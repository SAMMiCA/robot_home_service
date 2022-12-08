from typing import Optional, Set, Tuple, List, Dict, Any, Sequence, Union
import math
import torch
from allenact.base_abstractions.experiment_config import (
    ExperimentConfig,
    MachineParams,
    split_processes_onto_devices,
)
from experiments.home_service_il_base import HomeServiceILBaseExperimentConfig as Base


class HomeServiceClipRN50DaggerDistributedConfig(Base):
    def __init__(
        self,
        distributed_nodes: int = 1,
        num_devices: Optional[List[int]] = None,
        il_pipeline_types: Union[str, List[str]] = "20proc",
        expert_exploration_enabled: bool = False,
        include_other_move_actions: bool = True,
        screen_size: int = 224,
        training_steps: int = int(1e9),
        square_root_scaling: bool = True,
        rnn_type: str = "GRU",
        rnn_layers: int = 1,
    ) -> None:
        self.distributed_nodes = distributed_nodes
        self.NUM_DEVICES = num_devices if num_devices is not None else [torch.cuda.device_count()]
        self.IL_PIPELINE_TYPES = list(il_pipeline_types) if isinstance(il_pipeline_types, (List, Tuple, Set)) else [il_pipeline_types] * distributed_nodes
        self.EXPERT_EXPLORATION_ENABLED = expert_exploration_enabled
        self.INCLUDE_OTHER_MOVE_ACTIONS = include_other_move_actions
        self.SCREEN_SIZE = screen_size
        self.THOR_CONTROLLER_KWARGS = {
            "rotateStepDegrees": 90,
            "snapToGrid": True,
            "quality": "Very Low",
            "width": screen_size,
            "height": screen_size,
            "fastActionEmit": True,
            "scene": "Procedural",
        }
        self.TRAINING_STEPS = training_steps
        self.square_root_scaling = square_root_scaling
        self.RNN_TYPE = rnn_type
        self.RNN_LAYERS = rnn_layers if rnn_type == "GRU" else 2 * rnn_layers

    def tag(self) -> str:
        return f"HomeServiceClipRN50DaggerDistributed{self.distributed_nodes}_{self.IL_PIPELINE_TYPE}_{self.TRAINING_STEPS // int(1e6)}Msteps_{self.RNN_LAYERS}layers_{self.RNN_TYPE}"

    def machine_params(self, mode="train", **kwargs) -> MachineParams:
        assert (
            self.NUM_DEVICES is not None
            and isinstance(self.NUM_DEVICES, Sequence)
            and len(self.NUM_DEVICES) == self.distributed_nodes
        ), f"num_devices is None: {self.NUM_DEVICES is None} or len(num_devices) [{len(self.NUM_DEVICES)}] != self.distributed_nodes [{self.distributed_nodes}]"
        self.NUM_DEVICES = list(self.NUM_DEVICES)

        devices = sum(
            [
                list(range(min(self.NUM_DEVICES[id], int(self.IL_PIPELINE_TYPES[id].lower().split("proc")[0]))))
                for id in range(self.distributed_nodes)
            ], []
        )
        devices = tuple(torch.device(d) for d in devices)
        if mode == "train":
            nprocesses = [int(label.lower().split("proc")[0]) for label in self.IL_PIPELINE_TYPES]
            nprocesses = sum(
                [
                    split_processes_onto_devices(
                        nprocesses[id],
                        self.NUM_DEVICES[id],
                    )
                    for id in range(self.distributed_nodes)
                ], []
            )
            params = MachineParams(
                nprocesses=nprocesses,
                devices=devices,
                sampler_devices=devices,
                sensor_preprocessor_graph=self.create_preprocessor_graph(mode=mode)
            )

            if "machine_id" in kwargs:
                machine_id = kwargs["machine_id"]
                assert (
                    0 <= machine_id < self.distributed_nodes
                ), f"machine_id {machine_id} out of range [0, {self.distributed_nodes - 1}]"
            else:
                # Not a distributed configs
                assert self.distributed_nodes == 1, f"This must be a single node config."
                machine_id = 0
                
            assert (
                0 < self.NUM_DEVICES[machine_id] <= torch.cuda.device_count()
            ), f"num_device [{self.NUM_DEVICES[machine_id]}] out of range [1, {torch.cuda.device_count()}]"
            
            self.IL_PIPELINE_TYPE = self.IL_PIPELINE_TYPES[machine_id]
            local_worker_ids = list(
                range(
                    sum(self.NUM_DEVICES[:machine_id]),
                    sum(self.NUM_DEVICES[:machine_id + 1]),
                )
            )
            
            params.set_local_worker_ids(local_worker_ids)

        if mode == "valid":
            nprocesses = (0, )
            params = MachineParams(
                nprocesses=nprocesses,
                devices=devices,
                sampler_devices=None,
                sensor_preprocessor_graph=self.create_preprocessor_graph(mode=mode)
            )

        # print(
        #     f"devices {params.devices}"
        #     f"\nnprocesses {params.nprocesses}"
        #     f"\nsampler_devices {params.sampler_devices}"
        #     f"\nlocal_worker_ids {params.local_worker_ids}"
        # )

        return params

    def _use_label_to_get_training_params(self):
        params = super()._use_label_to_get_training_params()
        if self.square_root_scaling:
            params["lr"] *= math.sqrt(self.distributed_nodes)  # linear scaling
        else:
            params["lr"] *= self.distributed_nodes  # linear scaling
        return params
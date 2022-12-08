import copy
import platform
import os
import compress_json
import compress_pickle
from abc import abstractmethod
from typing import Optional, List, Sequence, Dict, Any, Tuple
from allenact.utils.misc_utils import md5_hash_str_as_int, partition_sequence

import gym.spaces
import stringcase
import torch
import torchvision.models
from torch import cuda, optim, nn
from torch.optim.lr_scheduler import LambdaLR

import ai2thor.platform
import datagen.datagen_utils as datagen_utils
from allenact.base_abstractions.experiment_config import (
    ExperimentConfig,
    MachineParams,
    split_processes_onto_devices,
)
from allenact.base_abstractions.sensor import ExpertActionSensor, SensorSuite, Sensor
from allenact.base_abstractions.preprocessor import Preprocessor, SensorPreprocessorGraph
from allenact.embodiedai.sensors.vision_sensors import DepthSensor, IMAGENET_RGB_MEANS, IMAGENET_RGB_STDS
from allenact.embodiedai.preprocessors.resnet import ResNetPreprocessor
from allenact.utils.experiment_utils import LinearDecay, TrainingPipeline, Builder
from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_util import get_open_x_displays
from env.baseline_models import HomeServiceActorCriticSimpleConvRNN, HomeServiceResNetActorCriticRNN

from env.constants import (
    FOV,
    PICKUPABLE_OBJECTS,
    OPENABLE_OBJECTS,
    RECEPTACLE_OBJECTS,
    SMOOTHING_FACTOR,
    STEP_SIZE,
    ROTATION_ANGLE,
    HORIZON_ANGLE,
    THOR_COMMIT_ID,
    PROCTHOR_COMMIT_ID,
    VISIBILITY_DISTANCE,
    STARTER_HOME_SERVICE_DATA_DIR,
)
from env.environment import HomeServiceMode
from env.sensors import RGBHomeServiceSensor, DepthHomeServiceSensor, RelativePositionChangeSensor, SubtaskHomeServiceSensor
from env.tasks import HomeServiceTaskSampler


def get_tasks(stage: str):
    task_names_file = os.path.join(
        STARTER_HOME_SERVICE_DATA_DIR,
        stage,
        "task_names.json.gz",
    )
    if os.path.exists(task_names_file):
        print(f"Cached tasks file found at {task_names_file}, using this file.")
        return compress_json.load(task_names_file)

    tasks = list(
        compress_pickle.load(
            os.path.join(
                STARTER_HOME_SERVICE_DATA_DIR,
                f'{stage}.pkl.gz'
            )
        ).keys()
    )
    tasks = [
        task.replace("train_", "").replace("val_", "").replace("test_", "")
        for task in tasks
    ]
    compress_json.dump(tasks, task_names_file)

    return tasks



class HomeServiceBaseExperimentConfig(ExperimentConfig):
    EXPERT_EXPLORATION_ENABLED = False

    # Task parameters
    MAX_STEPS = 500
    REQUIRE_DONE_ACTION = True
    FORCE_AXIS_ALIGNED_START = True
    RANDOMIZE_START_ROTATION_DURING_TRAINING = False
    SMOOTH_NAV = False
    SMOOTHING_FACTOR = SMOOTHING_FACTOR

    # Sensor info
    REFERENCE_SEGMENTATION = False
    SENSORS: Optional[Sequence[Sensor]] = None
    EGOCENTRIC_RGB_UUID = "rgb"
    EGOCENTRIC_RGB_RESNET_UUID = "rgb_resnet"
    DEPTH_UUID = "depth"

    # Environment parameters
    ENV_KWARGS = dict(mode=HomeServiceMode.SNAP,)
    SCREEN_SIZE = 224
    QUALITY = "Very Low"
    THOR_CONTROLLER_KWARGS = {
        "gridSize": STEP_SIZE,
        "visibilityDistance": VISIBILITY_DISTANCE,
        "rotateStepDegrees": ROTATION_ANGLE,
        "horizonStepDegrees": HORIZON_ANGLE,
        "smoothingFactor": SMOOTHING_FACTOR,
        "snapToGrid": True,
        "quality": QUALITY,
        "width": SCREEN_SIZE,
        "height": SCREEN_SIZE,
        "commit_id": PROCTHOR_COMMIT_ID,
        "fastActionEmit": True,
        "scene": "Procedural",
        "renderDepthImage": True,
        "renderSemanticSegmentation": REFERENCE_SEGMENTATION,
        "renderInstanceSegmentation": REFERENCE_SEGMENTATION,
    }
    HEADLESS = True
    INCLUDE_OTHER_MOVE_ACTIONS = True
    ORDERED_OBJECT_TYPES = list(sorted(PICKUPABLE_OBJECTS + RECEPTACLE_OBJECTS))
    FOV = FOV

    # YOLACT_KWARGS = YOLACT_KWARGS

    # Training parameters
    TRAINING_STEPS = int(25e6)
    SAVE_INTERVAL = int(1e5)
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING: Optional[Tuple[str, str]] = ("RN50", "clip")
    DEVICE = torch.device('cuda')

    # Model parameters
    PREV_ACTION_EMBEDDING_DIM: int = 32
    RNN_TYPE: str = "LSTM"
    NUM_RNN_LAYERS: int = 1
    HIDDEN_SIZE: int = 512

    RGB_NORMALIZATION = False
    DEPTH_NORMALIZATION = False

    # Actions
    PICKUP_ACTIONS = list(
        sorted(
            [
                f"pickup_{stringcase.snakecase(object_type)}"
                for object_type in PICKUPABLE_OBJECTS
            ]
        )
    )
    OPEN_ACTIONS = list(
        sorted(
            [
                f"open_by_type_{stringcase.snakecase(object_type)}"
                for object_type in OPENABLE_OBJECTS
            ]
        )
    )
    CLOSE_ACTIONS = list(
        sorted(
            [
                f"close_by_type_{stringcase.snakecase(object_type)}"
                for object_type in OPENABLE_OBJECTS
            ]
        )
    )
    PUT_ACTIONS = list(
        sorted(
            [
                f"put_{stringcase.snakecase(object_type)}"
                for object_type in RECEPTACLE_OBJECTS
            ]
        )
    )

    def actions(self):
        done_actions = (
            tuple()
            if not self.REQUIRE_DONE_ACTION
            else ("done",)
        )
        other_move_actions = (
            tuple()
            if not self.INCLUDE_OTHER_MOVE_ACTIONS
            else ("move_left", "move_right", "move_back",)
        )

        return (
            done_actions
            + (
                "move_ahead",
            )
            + other_move_actions
            + (
                "rotate_right",
                "rotate_left",
                "stand",
                "crouch",
                "look_up",
                "look_down",
                *self.PICKUP_ACTIONS,
                *self.OPEN_ACTIONS,
                *self.CLOSE_ACTIONS,
                *self.PUT_ACTIONS,
            )
        )

    def sensors(self) -> Sequence[Sensor]:
        mean, stdev = None, None
        if self.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is not None:
            cnn_type, pretraining_type = self.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING
            if pretraining_type.strip().lower() == "clip":
                from allenact_plugins.clip_plugin.clip_preprocessors import (
                    ClipResNetPreprocessor,
                )

                mean = ClipResNetPreprocessor.CLIP_RGB_MEANS
                stdev = ClipResNetPreprocessor.CLIP_RGB_STDS
            else:
                mean = IMAGENET_RGB_MEANS
                stdev = IMAGENET_RGB_STDS
        
        if mean is not None and stdev is not None:
            normalize = True
        else:
            normalize = False

        sensors = [
            RGBHomeServiceSensor(
                height=self.SCREEN_SIZE,
                width=self.SCREEN_SIZE,
                use_resnet_normalization=normalize,
                uuid=self.EGOCENTRIC_RGB_UUID,
                mean=mean,
                stdev=stdev,
            ),
            DepthHomeServiceSensor(
                height=self.SCREEN_SIZE,
                width=self.SCREEN_SIZE,
                use_normalization=False,
                uuid=self.DEPTH_UUID,
            ),
        ]

        return sensors
    
    def create_resnet_builder(self, in_uuid: str, out_uuid: str):
        if self.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is None:
            raise NotImplementedError
        cnn_type, pretraining_type = self.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING
        if pretraining_type == "imagenet":
            assert cnn_type in [
                "RN18",
                "RN50",
            ], "Only allow using RN18/RN50 with `imagenet` pretrained weights."
            return ResNetPreprocessor(
                input_height=self.THOR_CONTROLLER_KWARGS["height"],
                input_width=self.THOR_CONTROLLER_KWARGS["width"],
                output_width=7,
                output_height=7,
                output_dims=512 if "18" in cnn_type else 2048,
                pool=False,
                torchvision_resnet_model=getattr(
                    torchvision.models, f"resnet{cnn_type.replace('RN', '')}"
                ),
                input_uuids=[in_uuid],
                output_uuid=out_uuid,
                device=self.DEVICE
            )
        elif pretraining_type == "clip":
            from allenact_plugins.clip_plugin.clip_preprocessors import (
                ClipResNetPreprocessor,
            )
            import clip

            # Let's make sure we download the clip model now
            # so we don't download it on every spawned process
            clip.load(cnn_type, "cpu")

            return ClipResNetPreprocessor(
                rgb_input_uuid=in_uuid,
                clip_model_type=cnn_type,
                pool=False,
                output_uuid=out_uuid,
                device=self.DEVICE
            )
        else:
            raise NotImplementedError

    def preprocessors(self) -> Sequence[Preprocessor]:
        preprocessors = []
        if self.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is not None:
            preprocessors.append(
                self.create_resnet_builder(
                    in_uuid=self.EGOCENTRIC_RGB_UUID,
                    out_uuid=self.EGOCENTRIC_RGB_RESNET_UUID,
                )
            )
        
        return preprocessors

    def create_preprocessor_graph(self, mode: str) -> SensorPreprocessorGraph:
        additional_output_uuids = []

        return (
            None
            if len(self.preprocessors()) == 0
            else Builder(
                SensorPreprocessorGraph,
                {
                    "source_observation_spaces": SensorSuite(
                        [
                            sensor
                            for sensor in self.sensors()
                            if (mode == "train" or not isinstance(sensor, ExpertActionSensor))
                        ]
                    ).observation_spaces,
                    "preprocessors": self.preprocessors(),
                    "additional_output_uuids": additional_output_uuids,
                }
            )
        )

    def get_lr_scheduler_builder(self, use_lr_decay: bool):
        return (
            None
            if not use_lr_decay
            else Builder(
                LambdaLR,
                {
                    "lr_lambda": LinearDecay(
                        steps=self.TRAINING_STEPS // 3, startp=1.0, endp=1.0 / 3
                    )
                },
            )
        )

    def machine_params(self, mode="train", **kwargs) -> MachineParams:
        """Return the number of processes and gpu_ids to use with training."""
        num_gpus = cuda.device_count()
        has_gpu = num_gpus != 0

        sampler_devices = None
        if mode == "train":
            nprocesses = self.num_train_processes() if torch.cuda.is_available() else 1
            devices = (
                list(range(min(nprocesses, num_gpus)))
                if has_gpu
                else [torch.device("cpu")]
            )
        elif mode == "valid":
            devices = [num_gpus - 1] if has_gpu else [torch.device("cpu")]
            nprocesses = self.num_valid_processes() if has_gpu else 0
        else:
            nprocesses = self.num_test_processes() if has_gpu else 1
            devices = (
                list(range(min(nprocesses, num_gpus)))
                if has_gpu
                else [torch.device("cpu")]
            )

        nprocesses = split_processes_onto_devices(
            nprocesses=nprocesses, ndevices=len(devices)
        )

        return MachineParams(
            nprocesses=nprocesses,
            devices=devices,
            sampler_devices=sampler_devices,
            sensor_preprocessor_graph=self.create_preprocessor_graph(mode=mode),
        )

    def make_sampler_fn(
        self,
        stage: str,
        force_cache_reset: bool,
        seed: int,
        epochs: int,
        allowed_tasks: Optional[Sequence[str]] = None,
        allowed_pickup_objs: Optional[Sequence[str]] = None,
        allowed_target_receps: Optional[Sequence[str]] = None,
        x_display: Optional[str] = None,
        sensors: Optional[Sequence[Sensor]] = None,
        thor_controller_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> HomeServiceTaskSampler:
        sensors = self.sensors() if sensors is None else sensors
        if "mp_ctx" in kwargs:
            del kwargs["mp_ctx"]

        return HomeServiceTaskSampler.from_fixed_dataset(
            stage=stage,
            allowed_tasks=allowed_tasks,
            allowed_pickup_objs=allowed_pickup_objs,
            allowed_target_receps=allowed_target_receps,
            home_service_env_kwargs=dict(
                force_cache_reset=force_cache_reset,
                **self.ENV_KWARGS,
                controller_kwargs={
                    "x_display": x_display,
                    **self.THOR_CONTROLLER_KWARGS,
                    "renderDepthImage": any(
                        isinstance(s, DepthSensor) for s in sensors
                    ),
                    **(
                        {} if thor_controller_kwargs is None else thor_controller_kwargs
                    ),
                },
            ),
            seed=seed,
            sensors=SensorSuite(sensors),
            max_steps=self.MAX_STEPS,
            discrete_actions=self.actions(),
            smooth_nav=self.SMOOTH_NAV,
            smoothing_factor=self.SMOOTHING_FACTOR,
            require_done_action=self.REQUIRE_DONE_ACTION,
            force_axis_aligned_start=self.FORCE_AXIS_ALIGNED_START,
            epochs=epochs,
            expert_exploration_enabled=self.EXPERT_EXPLORATION_ENABLED,
            **kwargs,
        )

    def stagewise_task_sampler_args(
        self,
        stage: str,
        process_ind: int,
        total_processes: int,
        headless: bool = False,
        allowed_tasks: Optional[Sequence[str]] = None,
        allowed_pickup_objs: Optional[Sequence[str]] = None,
        allowed_target_receps: Optional[Sequence[str]] = None,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ):
        if allowed_tasks is not None:
            tasks = allowed_tasks
        else:
            tasks = get_tasks(stage=stage)

        allowed_tasks = list(
            sorted(partition_sequence(seq=tasks, parts=total_processes,)[process_ind])
        )

        seed = md5_hash_str_as_int(str(allowed_tasks))

        device = (
            devices[process_ind % len(devices)]
            if devices is not None and len(devices) > 0
            else torch.device("cpu")
        )

        x_display: Optional[str] = None
        gpu_device: Optional[int] = None
        thor_platform: Optional[ai2thor.platform.BaseLinuxPlatform] = None
        if self.HEADLESS or headless:
            gpu_device = device
            thor_platform = ai2thor.platform.CloudRendering

        elif platform.system() == "Linux":
            try:
                x_displays = get_open_x_displays(throw_error_if_empty=True)

                if devices is not None and len(
                    [d for d in devices if d != torch.device("cpu")]
                ) > len(x_displays):
                    get_logger().warning(
                        f"More GPU devices found than X-displays (devices: `{x_displays}`, x_displays: `{x_displays}`)."
                        f" This is not necessarily a bad thing but may mean that you're not using GPU memory as"
                        f" efficiently as possible. Consider following the instructions here:"
                        f" https://allenact.org/installation/installation-framework/#installation-of-ithor-ithor-plugin"
                        f" describing how to start an X-display on every GPU."
                    )
                x_display = x_displays[process_ind % len(x_displays)]
            except IOError:
                # Could not find an open `x_display`, use CloudRendering instead.
                assert all(
                    [d != torch.device("cpu") and d >= 0 for d in devices]
                ), "Cannot use CPU devices when there are no open x-displays as CloudRendering requires specifying a GPU."
                gpu_device = device
                thor_platform = ai2thor.platform.CloudRendering

        kwargs = {
            "stage": stage,
            "allowed_tasks": allowed_tasks,
            "allowed_pickup_objs": allowed_pickup_objs,
            "allowed_target_receps": allowed_target_receps,
            "seed": seed,
            "x_display": x_display,
            "thor_controller_kwargs": {
                "gpu_device": gpu_device,
                "platform": thor_platform,
            },
        }
        sensors = kwargs.get("sensors", copy.deepcopy(self.sensors()))
        kwargs["sensors"] = sensors

        return kwargs

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ):
        return dict(
            force_cache_reset=False,
            epochs=float("inf"),
            task_spec_in_metrics=True,
            **self.stagewise_task_sampler_args(
                stage="train",
                process_ind=process_ind,
                total_processes=total_processes,
                devices=devices,
                seeds=seeds,
                deterministic_cudnn=deterministic_cudnn,
            ),
        )

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ):
        return dict(
            force_cache_reset=True,
            epochs=1,
            **self.stagewise_task_sampler_args(
                stage="val",
                process_ind=process_ind,
                total_processes=total_processes,
                devices=devices,
                seeds=seeds,
                deterministic_cudnn=deterministic_cudnn,
            ),
        )

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
        task_spec_in_metrics: bool = False,
    ):
        task_spec_in_metrics = False

        stage = "test"

        return dict(
            force_cache_reset=True,
            epochs=1,
            task_spec_in_metrics=task_spec_in_metrics,
            **self.stagewise_task_sampler_args(
                stage=stage,
                process_ind=process_ind,
                total_processes=total_processes,
                devices=devices,
                seeds=seeds,
                deterministic_cudnn=deterministic_cudnn,
            ),
        )

    @abstractmethod
    def _training_pipeline_info(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def num_train_processes(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def num_valid_processes(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def num_test_processes(self) -> int:
        raise NotImplementedError

    def training_pipeline(self, **kwargs) -> TrainingPipeline:
        info = self._training_pipeline_info()

        return TrainingPipeline(
            gamma=info.get("gamma", 0.99),
            use_gae=info.get("use_gae", True),
            gae_lambda=info.get("gae_lambda", 0.95),
            num_steps=info["num_steps"],
            num_mini_batch=info["num_mini_batch"],
            update_repeats=info["update_repeats"],
            max_grad_norm=info.get("max_grad_norm", 0.5),
            save_interval=self.SAVE_INTERVAL,
            named_losses=info["named_losses"],
            metric_accumulate_interval=self.num_train_processes()
            * self.MAX_STEPS
            if torch.cuda.is_available()
            else 1,
            optimizer_builder=Builder(optim.Adam, dict(lr=info["lr"])),
            advance_scene_rollout_period=None,
            pipeline_stages=info["pipeline_stages"],
            lr_scheduler_builder=self.get_lr_scheduler_builder(
                use_lr_decay=info["use_lr_decay"]
            ),
        )

    def create_model(self, **kwargs) -> nn.Module:
        if self.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is None:
            return HomeServiceActorCriticSimpleConvRNN(
                action_space=gym.spaces.Discrete(len(self.actions())),
                observation_space=SensorSuite(self.sensors()).observation_spaces,
                rgb_uuid=self.EGOCENTRIC_RGB_UUID,
                ordered_object_types=self.ORDERED_OBJECT_TYPES
            )
        else:
            return HomeServiceResNetActorCriticRNN(
                action_space=gym.spaces.Discrete(len(self.actions())),
                observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
                rgb_uuid=self.EGOCENTRIC_RGB_RESNET_UUID,
                ordered_object_types=self.ORDERED_OBJECT_TYPES
            )
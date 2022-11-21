import copy
import platform
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
    YOLACT_KWARGS,
)
from env.environment import HomeServiceMode
from env.sensors import RGBHomeServiceSensor, DepthHomeServiceSensor, RelativePositionChangeSensor, SubtaskHomeServiceSensor
from env.tasks import HomeServiceTaskSampler


class HomeServiceBaseExperimentConfig(ExperimentConfig):

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
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING: Optional[Tuple[str, str]] = None
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

    @classmethod
    def actions(cls):
        done_actions = (
            tuple()
            if not cls.REQUIRE_DONE_ACTION
            else ("done",)
        )
        other_move_actions = (
            tuple()
            if not cls.INCLUDE_OTHER_MOVE_ACTIONS
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
                *cls.PICKUP_ACTIONS,
                *cls.OPEN_ACTIONS,
                *cls.CLOSE_ACTIONS,
                *cls.PUT_ACTIONS,
            )
        )

    @classmethod
    def sensors(cls):
        mean, stdev = None, None
        if cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is not None:
            cnn_type, pretraining_type = cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING
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
                height=cls.SCREEN_SIZE,
                width=cls.SCREEN_SIZE,
                use_resnet_normalization=normalize,
                uuid=cls.EGOCENTRIC_RGB_UUID,
                mean=mean,
                stdev=stdev,
            ),
            DepthHomeServiceSensor(
                height=cls.SCREEN_SIZE,
                width=cls.SCREEN_SIZE,
                use_normalization=False,
                uuid=cls.DEPTH_UUID,
            ),
        ]

        return sensors
    
    @ classmethod
    def create_resnet_builder(cls, in_uuid: str, out_uuid: str):
        if cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is None:
            raise NotImplementedError
        cnn_type, pretraining_type = cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING
        if pretraining_type == "imagenet":
            assert cnn_type in [
                "RN18",
                "RN50",
            ], "Only allow using RN18/RN50 with `imagenet` pretrained weights."
            return ResNetPreprocessor(
                input_height=cls.THOR_CONTROLLER_KWARGS["height"],
                input_width=cls.THOR_CONTROLLER_KWARGS["width"],
                output_width=7,
                output_height=7,
                output_dims=512 if "18" in cnn_type else 2048,
                pool=False,
                torchvision_resnet_model=getattr(
                    torchvision.models, f"resnet{cnn_type.replace('RN', '')}"
                ),
                input_uuids=[in_uuid],
                output_uuid=out_uuid,
                device=cls.DEVICE
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
                device=DEVICE
            )
        else:
            raise NotImplementedError

    @classmethod
    def preprocessors(cls) -> Sequence[Preprocessor]:
        preprocessors = []
        if cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is not None:
            preprocessors.append(
                cls.create_resnet_builder(
                    in_uuid=cls.EGOCENTRIC_RGB_UUID,
                    out_uuid=cls.EGOCENTRIC_RGB_RESNET_UUID,
                )
            )
        
        return preprocessors

    @classmethod
    def create_preprocessor_graph(cls, mode: str) -> SensorPreprocessorGraph:
        additional_output_uuids = []

        return (
            None
            if len(cls.preprocessors()) == 0
            else Builder(
                SensorPreprocessorGraph,
                {
                    "source_observation_spaces": SensorSuite(cls.sensors()).observation_spaces,
                    "preprocessors": cls.preprocessors(),
                    "additional_output_uuids": additional_output_uuids,
                }
            )
        )

    
    @classmethod
    def get_lr_scheduler_builder(cls, use_lr_decay: bool):
        return (
            None
            if not use_lr_decay
            else Builder(
                LambdaLR,
                {
                    "lr_lambda": LinearDecay(
                        steps=cls.TRAINING_STEPS // 3, startp=1.0, endp=1.0 / 3
                    )
                },
            )
        )

    @classmethod
    def machine_params(cls, mode="train", **kwargs) -> MachineParams:
        """Return the number of processes and gpu_ids to use with training."""
        num_gpus = cuda.device_count()
        has_gpu = num_gpus != 0

        sampler_devices = None
        if mode == "train":
            nprocesses = cls.num_train_processes() if torch.cuda.is_available() else 1
            devices = (
                list(range(min(nprocesses, num_gpus)))
                if has_gpu
                else [torch.device("cpu")]
            )
        elif mode == "valid":
            devices = [num_gpus - 1] if has_gpu else [torch.device("cpu")]
            nprocesses = cls.num_valid_processes() if has_gpu else 0
        else:
            nprocesses = cls.num_test_processes() if has_gpu else 1
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
            sensor_preprocessor_graph=cls.create_preprocessor_graph(mode=mode),
        )


    @classmethod
    def make_sampler_fn(
        cls,
        stage: str,
        force_cache_reset: bool,
        seed: int,
        allowed_task_keys: Optional[Sequence[str]] = None,
        allowed_pickup_objs: Optional[Sequence[str]] = None,
        allowed_start_receps: Optional[Sequence[str]] = None,
        allowed_target_receps: Optional[Sequence[str]] = None,
        allowed_scene_inds: Optional[Sequence[int]] = None,
        x_display: Optional[str] = None,
        sensors: Optional[Sequence[Sensor]] = None,
        thor_controller_kwargs: Optional[Dict] = None,
        runtime_sample: bool = False,
        repeats_before_scene_change: int = 1,
        **kwargs,
    ) -> HomeServiceTaskSampler:
        sensors = cls.sensors(stage) if sensors is None else sensors
        if "mp_ctx" in kwargs:
            del kwargs["mp_ctx"]

        if not runtime_sample:
            return HomeServiceTaskSampler.from_fixed_simple_pick_and_place_data(
                stage=stage,
                allowed_task_keys=allowed_task_keys,
                allowed_pickup_objs=allowed_pickup_objs,
                allowed_start_receps=allowed_start_receps,
                allowed_target_receps=allowed_target_receps,
                allowed_scene_inds=allowed_scene_inds,
                home_service_env_kwargs=dict(
                    force_cache_reset=force_cache_reset,
                    **cls.ENV_KWARGS,
                    controller_kwargs={
                        "x_display": x_display,
                        **cls.THOR_CONTROLLER_KWARGS,
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
                max_steps=cls.MAX_STEPS,
                discrete_actions=cls.actions(),
                smooth_nav=cls.SMOOTH_NAV,
                smoothing_factor=cls.SMOOTHING_FACTOR,
                require_done_action=cls.REQUIRE_DONE_ACTION,
                force_axis_aligned_start=cls.FORCE_AXIS_ALIGNED_START,
                randomize_start_rotation=(stage == "train"
                and cls.RANDOMIZE_START_ROTATION_DURING_TRAINING),
                **kwargs,
            )

    @classmethod
    def stagewise_task_sampler_args(
        cls,
        stage: str,
        process_ind: int,
        total_processes: int,
        headless: bool = False,
        allowed_pickup_objs: Optional[Sequence[str]] = None,
        allowed_start_receps: Optional[Sequence[str]] = None,
        allowed_target_receps: Optional[Sequence[str]] = None,
        allowed_scene_inds: Optional[Sequence[int]] = None,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ):
        task_keys = datagen_utils.get_task_keys(stage)

        allowed_task_keys = list(
            sorted(partition_sequence(seq=task_keys, parts=total_processes,)[process_ind])
        )

        seed = md5_hash_str_as_int(str(task_keys))

        device = (
            devices[process_ind % len(devices)]
            if devices is not None and len(devices) > 0
            else torch.device("cpu")
        )

        if allowed_scene_inds is None:
            allowed_scene_inds = datagen_utils.get_scene_inds(stage)
        else:
            assert all([scene_ind in datagen_utils.get_scene_inds(stage) for scene_ind in allowed_scene_inds])

        x_display: Optional[str] = None
        gpu_device: Optional[int] = None
        thor_platform: Optional[ai2thor.platform.BaseLinuxPlatform] = None
        if cls.HEADLESS:
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
            "allowed_task_keys": allowed_task_keys,
            "allowed_pickup_objs": allowed_pickup_objs,
            "allowed_start_receps": allowed_start_receps,
            "allowed_target_receps": allowed_target_receps,
            "allowed_scene_inds": allowed_scene_inds,
            "seed": seed,
            "x_display": x_display,
            "thor_controller_kwargs": {
                "gpu_device": gpu_device,
                "platform": thor_platform,
            },
        }
        sensors = kwargs.get("sensors", copy.deepcopy(cls.sensors()))
        kwargs["sensors"] = sensors

        return kwargs

    @classmethod
    def train_task_sampler_args(
        cls,
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
            **cls.stagewise_task_sampler_args(
                stage="train_seen",
                process_ind=process_ind,
                total_processes=total_processes,
                devices=devices,
                seeds=seeds,
                deterministic_cudnn=deterministic_cudnn,
            ),
        )

    @classmethod
    def valid_task_sampler_args(
        cls,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ):
        return dict(
            force_cache_reset=True,
            epochs=1,
            **cls.stagewise_task_sampler_args(
                stage="train_seen",
                # allowed_pickup_objs=None,
                # allowed_start_receps=None,
                # allowed_target_receps=None,
                allowed_scene_inds=(10, 20),
                process_ind=process_ind,
                total_processes=total_processes,
                devices=devices,
                seeds=seeds,
                deterministic_cudnn=deterministic_cudnn,
            ),
        )

    @classmethod
    def test_task_sampler_args(
        cls,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
        task_spec_in_metrics: bool = False,
    ):
        task_spec_in_metrics = False

        stage = "train_unseen"

        return dict(
            force_cache_reset=True,
            epochs=1,
            task_spec_in_metrics=task_spec_in_metrics,
            **cls.stagewise_task_sampler_args(
                stage=stage,
                # allowed_pickup_objs=None,
                # allowed_start_receps=None,
                # allowed_target_receps=None,
                allowed_scene_inds=range(29, 31),
                process_ind=process_ind,
                total_processes=total_processes,
                devices=devices,
                seeds=seeds,
                deterministic_cudnn=deterministic_cudnn,
            ),
        )

    @classmethod
    @abstractmethod
    def _training_pipeline_info(cls) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def num_train_processes(cls) -> int:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def num_valid_processes(cls) -> int:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def num_test_processes(cls) -> int:
        raise NotImplementedError

    @classmethod
    def training_pipeline(cls, **kwargs) -> TrainingPipeline:
        info = cls._training_pipeline_info()

        return TrainingPipeline(
            gamma=info.get("gamma", 0.99),
            use_gae=info.get("use_gae", True),
            gae_lambda=info.get("gae_lambda", 0.95),
            num_steps=info["num_steps"],
            num_mini_batch=info["num_mini_batch"],
            update_repeats=info["update_repeats"],
            max_grad_norm=info.get("max_grad_norm", 0.5),
            save_interval=cls.SAVE_INTERVAL,
            named_losses=info["named_losses"],
            metric_accumulate_interval=cls.num_train_processes()
            * cls.MAX_STEPS
            if torch.cuda.is_available()
            else 1,
            optimizer_builder=Builder(optim.Adam, dict(lr=info["lr"])),
            advance_scene_rollout_period=None,
            pipeline_stages=info["pipeline_stages"],
            lr_scheduler_builder=cls.get_lr_scheduler_builder(
                use_lr_decay=info["use_lr_decay"]
            ),
        )

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        if cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is None:
            return HomeServiceActorCriticSimpleConvRNN(
                action_space=gym.spaces.Discrete(len(cls.actions())),
                observation_space=SensorSuite(cls.sensors()).observation_spaces,
                rgb_uuid=cls.EGOCENTRIC_RGB_UUID,
                subtask_uuid="subtask",
                rel_position_change_uuid="rel_position_change",
                ordered_object_types=cls.ORDERED_OBJECT_TYPES
            )
        else:
            return HomeServiceResNetActorCriticRNN(
                action_space=gym.spaces.Discrete(len(cls.actions())),
                observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
                rgb_uuid=cls.EGOCENTRIC_RGB_RESNET_UUID,
                subtask_uuid="subtask",
                rel_position_change_uuid="rel_position_change",
                ordered_object_types=cls.ORDERED_OBJECT_TYPES
            )
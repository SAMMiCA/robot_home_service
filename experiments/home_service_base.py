import copy
import platform
from abc import abstractmethod
from typing import Optional, List, Sequence, Dict, Any
from allenact.utils.misc_utils import md5_hash_str_as_int, partition_sequence

import gym.spaces
import stringcase
import torch
import torchvision.models
from torch import cuda, optim, nn
from torch.optim.lr_scheduler import LambdaLR

import datagen.datagen_utils as datagen_utils
from allenact.base_abstractions.experiment_config import (
    ExperimentConfig,
    MachineParams,
    split_processes_onto_devices,
)
from allenact.base_abstractions.sensor import ExpertActionSensor, SensorSuite, Sensor
from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph
from allenact.embodiedai.sensors.vision_sensors import DepthSensor
from allenact.embodiedai.preprocessors.resnet import ResNetPreprocessor
from allenact.utils.experiment_utils import LinearDecay, TrainingPipeline, Builder
from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_util import get_open_x_displays
from env.baseline_models import HomeServiceActorCriticSimpleConvRNN, HomeServiceResNetActorCriticRNN

from env.constants import (
    FOV,
    LOCAL_EXECUTABLE_PATH,
    PICKUPABLE_OBJECTS,
    OPENABLE_OBJECTS,
    RECEPTACLE_OBJECTS,
    SMOOTHING_FACTOR,
    STEP_SIZE,
    ROTATION_ANGLE,
    HORIZON_ANGLE,
    THOR_COMMIT_ID,
    VISIBILITY_DISTANCE,
    YOLACT_KWARGS,
)
from env.environment import HomeServiceMode
from env.sensors import RGBHomeServiceSensor, DepthHomeServiceSensor, RelativePositionChangeSensor, SubtaskHomeServiceSensor
from env.tasks import HomeServiceTaskSampler, HomeServiceTaskType


class HomeServiceBaseExperimentConfig(ExperimentConfig):

    # Task parameters
    MAX_STEPS = 500
    REQUIRE_DONE_ACTION = True
    REQUIRE_PASS_ACTION = True
    REQUIRE_GOTO_ACTION = True
    FORCE_AXIS_ALIGNED_START = True
    RANDOMIZE_START_ROTATION_DURING_TRAINING = False
    SMOOTH_NAV = True
    SMOOTHING_FACTOR = SMOOTHING_FACTOR

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
        "commit_id": THOR_COMMIT_ID,
        # "local_executable_path": LOCAL_EXECUTABLE_PATH,
        "fastActionEmit": True,
        "renderDepthImage": False,
        "renderSemanticSegmentation": False,
        "renderInstanceSegmentation": False,
    }
    INCLUDE_OTHER_MOVE_ACTIONS = False
    ORDERED_OBJECT_TYPES = list(sorted(PICKUPABLE_OBJECTS + RECEPTACLE_OBJECTS))
    FOV = FOV

    # YOLACT_KWARGS = YOLACT_KWARGS

    # Training parameters
    TRAINING_STEPS = int(25e6)
    SAVE_INTERVAL = int(1e5)
    USE_RESNET_CNN = False

    # Sensor info
    SENSORS: Optional[Sequence[Sensor]] = None
    EGOCENTRIC_RGB_UUID = "rgb"
    EGOCENTRIC_RGB_RESNET_UUID = "rgb_resnet"
    DEPTH_UUID = "depth"

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
                f"put_by_type_{stringcase.snakecase(object_type)}"
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
        pass_actions = (
            tuple()
            if not cls.REQUIRE_PASS_ACTION
            else ("pass",)
        )
        goto_actions = (
            tuple()
            if not cls.REQUIRE_GOTO_ACTION
            else ("goto_kitchen", "goto_living_room", "goto_bedroom", "goto_bathroom")
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
                # *cls.PICKUP_ACTIONS,
                # *cls.OPEN_ACTIONS,
                # *cls.CLOSE_ACTIONS,
                # *cls.PUT_ACTIONS,
                "pickup",
                # "open",
                # "close",
                "put",                
            )
            + goto_actions
            + pass_actions
        )

    @classmethod
    def sensors(cls, mode):
        sensors = [
            RGBHomeServiceSensor(
                height=cls.SCREEN_SIZE,
                width=cls.SCREEN_SIZE,
                use_resnet_normalization=True if cls.USE_RESNET_CNN else False,
                uuid=cls.EGOCENTRIC_RGB_UUID,
            ),
            DepthHomeServiceSensor(
                height=cls.SCREEN_SIZE,
                width=cls.SCREEN_SIZE,
                use_normalization=False,
                uuid=cls.DEPTH_UUID,
            ),
            RelativePositionChangeSensor(
                base=ROTATION_ANGLE,
            ),
            SubtaskHomeServiceSensor(
                object_types=cls.ORDERED_OBJECT_TYPES,
            ),
            # YolactObjectDetectionSensor(
            #     ordered_object_types=cls.ORDERED_OBJECT_TYPES,
            #     **cls.YOLACT_KWARGS
            # )
        ]

        return sensors
    
    @ classmethod
    def resnet_preprocessor_graph(cls, mode: str) -> SensorPreprocessorGraph:
        def create_resnet_builder(in_uuid: str, out_uuid: str):
            return ResNetPreprocessor(
                input_height=cls.THOR_CONTROLLER_KWARGS["height"],
                input_width=cls.THOR_CONTROLLER_KWARGS["width"],
                output_width=7,
                output_height=7,
                output_dims=512,
                pool=False,
                torchvision_resnet_model=torchvision.models.resnet18,
                input_uuids=[in_uuid],
                output_uuid=out_uuid,
            )

        img_uuids = [cls.EGOCENTRIC_RGB_UUID]
        return SensorPreprocessorGraph(
            source_observation_spaces=SensorSuite(
                [
                    sensor
                    for sensor in cls.sensors(mode)
                    if (mode == "train" or not isinstance(sensor, ExpertActionSensor))
                ]
            ).observation_spaces,
            preprocessors=[
                create_resnet_builder(sid, f"{sid}_resnet") for sid in img_uuids
            ],
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
            sensor_preprocessor_graph=cls.resnet_preprocessor_graph(mode=mode)
            if cls.USE_RESNET_CNN
            else None,
        )


    @classmethod
    def make_sampler_fn(
        cls,
        stage: str,
        force_cache_reset: bool,
        # allowed_scenes: Optional[Sequence[str]],
        seed: int,
        task_type: HomeServiceTaskType,
        allowed_task_keys: Optional[Sequence[str]] = None,
        allowed_pickup_objs: Optional[Sequence[str]] = None,
        allowed_start_receps: Optional[Sequence[str]] = None,
        allowed_target_receps: Optional[Sequence[str]] = None,
        allowed_scene_inds: Optional[Sequence[int]] = None,
        # scene_to_allowed_inds: Optional[Dict[str, Sequence[int]]] = None,
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
                # allowed_scenes=allowed_scenes,
                # scene_to_allowed_inds=scene_to_allowed_inds,
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
                task_type=task_type,
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
        else:
            # return HomeServiceTaskSampler.from_scenes_at_runtime(
            #     stage=stage,
            #     allowed_scenes=allowed_scenes,
            #     repeats_before_scene_change=repeats_before_scene_change,
            #     home_service_env_kwargs=dict(
            #         force_cache_reset=force_cache_reset,
            #         **cls.ENV_KWARGS,
            #         controller_kwargs={
            #             "x_display": x_display,
            #             **cls.THOR_CONTROLLER_KWARGS,
            #             "renderDepthImage": any(
            #                 isinstance(s, DepthSensor) for s in sensors
            #             ),
            #             **(
            #                 {} if thor_controller_kwargs is None else thor_controller_kwargs
            #             ),
            #         },
            #     ),
            #     seed=seed,
            #     task_type=task_type,
            #     sensors=SensorSuite(sensors),
            #     max_steps=cls.MAX_STEPS,
            #     discrete_actions=cls.actions(),
            #     smooth_nav=cls.SMOOTH_NAV,
            #     smoothing_factor=cls.SMOOTHING_FACTOR,
            #     require_done_action=cls.REQUIRE_DONE_ACTION,
            #     force_axis_aligned_start=cls.FORCE_AXIS_ALIGNED_START,
            #     **kwargs,
            # )
            pass

    # @classmethod
    # def stagewise_task_sampler_args(
    #     cls,
    #     stage: str,
    #     process_ind: int,
    #     total_processes: int,
    #     headless: bool = False,
    #     allowed_inds_subset: Optional[Sequence[int]] = None,
    #     devices: Optional[List[int]] = None,
    #     seeds: Optional[List[int]] = None,
    #     deterministic_cudnn: bool = False,
    # ):
    #     if stage == "combined":

    #         train_scenes = datagen_utils.get_scenes("train")
    #         other_scenes = datagen_utils.get_scenes("val") + datagen_utils.get_scenes("test")

    #         assert len(train_scenes) == 2 * len(other_scenes)
    #         scenes = []

    #         while len(train_scenes) != 0:
    #             scenes.append(train_scenes.pop())
    #             scenes.append(train_scenes.pop())
    #             scenes.append(other_scenes.pop())
    #         assert len(train_scenes) == len(other_scenes)
        
    #     else:
    #         scenes = datagen_utils.get_scenes(stage)

    #     if total_processes > len(scenes):
    #         assert stage == "train" and total_processes % len(scenes) == 0
    #         scenes = scenes * (total_processes // len(scenes))

    #     allowed_scenes = list(
    #         sorted(partition_sequence(seq=scenes, parts=total_processes,)[process_ind])
    #     )

    #     scene_to_allowed_inds = None
    #     if allowed_inds_subset is not None:
    #         allowed_inds_subset = tuple(allowed_inds_subset)
    #         # assert stage in ["valid", "train_unseen"]
    #         scene_to_allowed_inds = {
    #             scene: allowed_inds_subset for scene in allowed_scenes
    #         }
    #     seed = md5_hash_str_as_int(str(allowed_scenes))

    #     device = (
    #         devices[process_ind % len(devices)]
    #         if devices is not None and len(devices) > 0
    #         else torch.device("cpu")
    #     )

    #     x_display: Optional[str] = None
    #     if headless:
    #         if platform.system() == "Linux":
    #             x_displays = get_open_x_displays(throw_error_if_empty=True)

    #             if devices is not None and len(
    #                 [d for d in devices if d != torch.device("cpu")]
    #             ) > len(x_displays):
    #                 get_logger().warning(
    #                     f"More GPU devices found than X-displays (devices: `{x_displays}`, x_displays: `{x_displays}`)."
    #                     f" This is not necessarily a bad thing but may mean that you're not using GPU memory as"
    #                     f" efficiently as possible. Consider following the instructions here:"
    #                     f" https://allenact.org/installation/installation-framework/#installation-of-ithor-ithor-plugin"
    #                     f" describing how to start an X-display on every GPU."
    #                 )
    #             x_display = x_displays[process_ind % len(x_displays)]

    #     kwargs = {
    #         "stage": stage,
    #         "allowed_scenes": allowed_scenes,
    #         "scene_to_allowed_inds": scene_to_allowed_inds,
    #         "seed": seed,
    #         "x_display": x_display,
    #     }
    #     sensors = kwargs.get("sensors", copy.deepcopy(cls.sensors(stage)))
    #     kwargs["sensors"] = sensors

    #     return kwargs

    # @classmethod
    # def train_task_sampler_args(
    #     cls,
    #     process_ind: int,
    #     total_processes: int,
    #     headless: bool = False,
    #     devices: Optional[List[int]] = None,
    #     seeds: Optional[List[int]] = None,
    #     deterministic_cudnn: bool = False,
    # ):
    #     return dict(
    #         force_cache_reset=False,
    #         epochs=float("inf"),
    #         **cls.stagewise_task_sampler_args(
    #             stage="train",
    #             process_ind=process_ind,
    #             total_processes=total_processes,
    #             headless=headless,
    #             devices=devices,
    #             seeds=seeds,
    #             deterministic_cudnn=deterministic_cudnn,
    #         ),
    #     )

    # @classmethod
    # def valid_task_sampler_args(
    #     cls,
    #     process_ind: int,
    #     total_processes: int,
    #     headless: bool = False,
    #     devices: Optional[List[int]] = None,
    #     seeds: Optional[List[int]] = None,
    #     deterministic_cudnn: bool = False,
    # ):
    #     return dict(
    #         force_cache_reset=True,
    #         epochs=1,
    #         **cls.stagewise_task_sampler_args(
    #             stage="valid",
    #             allowed_inds_subset=tuple(range(10)),
    #             process_ind=process_ind,
    #             total_processes=total_processes,
    #             headless=headless,
    #             devices=devices,
    #             seeds=seeds,
    #             deterministic_cudnn=deterministic_cudnn,
    #         ),
    #     )

    # @classmethod
    # def test_task_sampler_args(
    #     cls,
    #     process_ind: int,
    #     total_processes: int,
    #     headless: bool = False,
    #     devices: Optional[List[int]] = None,
    #     seeds: Optional[List[int]] = None,
    #     deterministic_cudnn: bool = False,
    #     task_spec_in_metrics: bool = False,
    # ):
    #     task_spec_in_metrics = False

    #     stage = "combined"
    #     allowed_inds_subset = None

    #     return dict(
    #         force_cache_reset=True,
    #         epochs=1,
    #         task_spec_in_metrics=task_spec_in_metrics,
    #         **cls.stagewise_task_sampler_args(
    #             stage=stage,
    #             allowed_inds_subset=allowed_inds_subset,
    #             process_ind=process_ind,
    #             total_processes=total_processes,
    #             headless=headless,
    #             devices=devices,
    #             seeds=seeds,
    #             deterministic_cudnn=deterministic_cudnn,
    #         ),
    #     )

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
        if headless:
            if platform.system() == "Linux":
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

        kwargs = {
            "stage": stage,
            "allowed_task_keys": allowed_task_keys,
            "allowed_pickup_objs": allowed_pickup_objs,
            "allowed_start_receps": allowed_start_receps,
            "allowed_target_receps": allowed_target_receps,
            "allowed_scene_inds": allowed_scene_inds,
            "seed": seed,
            "x_display": x_display,
        }
        sensors = kwargs.get("sensors", copy.deepcopy(cls.sensors(stage)))
        kwargs["sensors"] = sensors

        return kwargs

    @classmethod
    def train_task_sampler_args(
        cls,
        process_ind: int,
        total_processes: int,
        headless: bool = False,
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
                headless=headless,
                # allowed_pickup_objs=None,
                # allowed_start_receps=None,
                # allowed_target_receps=None,
                # allowed_scene_inds=None,
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
        headless: bool = False,
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
                headless=headless,
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
        headless: bool = False,
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
                headless=headless,
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
        if not cls.USE_RESNET_CNN:
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
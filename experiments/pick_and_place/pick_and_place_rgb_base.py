

from typing import Dict, Optional, Sequence

from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact.embodiedai.sensors.vision_sensors import DepthSensor
from env.tasks import HomeServiceTaskSampler, HomeServiceTaskType
from experiments.home_service_base import HomeServiceBaseExperimentConfig


class PickAndPlaceRGBBaseExperimentConfig(HomeServiceBaseExperimentConfig):

    # @classmethod
    # def make_sampler_fn(
    #     cls,
    #     stage: str,
    #     force_cache_reset: bool,
    #     allowed_scenes: Optional[Sequence[str]],
    #     seed: int,
    #     scene_to_allowed_inds: Optional[Dict[str, Sequence[int]]] = None,
    #     x_display: Optional[str] = None,
    #     sensors: Optional[Sequence[Sensor]] = None,
    #     thor_controller_kwargs: Optional[Dict] = None,
    #     runtime_sample: bool = False,
    #     repeats_before_scene_change: int = 1,
    #     **kwargs,
    # ) -> HomeServiceTaskSampler:
    #     sensors = cls.sensors(stage) if sensors is None else sensors
    #     if "mp_ctx" in kwargs:
    #         del kwargs["mp_ctx"]

    #     if not runtime_sample:
    #         return HomeServiceTaskSampler.from_fixed_dataset(
    #             stage=stage,
    #             allowed_scenes=allowed_scenes,
    #             scene_to_allowed_inds=scene_to_allowed_inds,
    #             home_service_env_kwargs=dict(
    #                 force_cache_reset=force_cache_reset,
    #                 **cls.ENV_KWARGS,
    #                 controller_kwargs={
    #                     "x_display": x_display,
    #                     **cls.THOR_CONTROLLER_KWARGS,
    #                     "renderDepthImage": any(
    #                         isinstance(s, DepthSensor) for s in sensors
    #                     ),
    #                     **(
    #                         {} if thor_controller_kwargs is None else thor_controller_kwargs
    #                     ),
    #                 },
    #             ),
    #             seed=seed,
    #             task_type=HomeServiceTaskType.SIMPLE_PICK_AND_PLACE,
    #             sensors=SensorSuite(sensors),
    #             max_steps=cls.MAX_STEPS,
    #             discrete_actions=cls.actions(),
    #             smooth_nav=cls.SMOOTH_NAV,
    #             smoothing_factor=cls.SMOOTHING_FACTOR,
    #             require_done_action=cls.REQUIRE_DONE_ACTION,
    #             force_axis_aligned_start=cls.FORCE_AXIS_ALIGNED_START,
    #             randomize_start_rotation=stage == "train"
    #             and cls.RANDOMIZE_START_ROTATION_DURING_TRAINING,
    #             **kwargs,
    #         )
    #     else:
    #         return HomeServiceTaskSampler.from_scenes_at_runtime(
    #             stage=stage,
    #             allowed_scenes=allowed_scenes,
    #             repeats_before_scene_change=repeats_before_scene_change,
    #             home_service_env_kwargs=dict(
    #                 force_cache_reset=force_cache_reset,
    #                 **cls.ENV_KWARGS,
    #                 controller_kwargs={
    #                     "x_display": x_display,
    #                     **cls.THOR_CONTROLLER_KWARGS,
    #                     "renderDepthImage": any(
    #                         isinstance(s, DepthSensor) for s in sensors
    #                     ),
    #                     **(
    #                         {} if thor_controller_kwargs is None else thor_controller_kwargs
    #                     ),
    #                 },
    #             ),
    #             seed=seed,
    #             task_type=HomeServiceTaskType.SIMPLE_PICK_AND_PLACE,
    #             sensors=SensorSuite(sensors),
    #             max_steps=cls.MAX_STEPS,
    #             discrete_actions=cls.actions(),
    #             smooth_nav=cls.SMOOTH_NAV,
    #             smoothing_factor=cls.SMOOTHING_FACTOR,
    #             require_done_action=cls.REQUIRE_DONE_ACTION,
    #             force_axis_aligned_start=cls.FORCE_AXIS_ALIGNED_START,
    #             **kwargs,
    #         )
    @classmethod
    def make_sampler_fn(
        cls,
        **init_kwargs,
    ) -> HomeServiceTaskSampler:
        if "task_type" in init_kwargs:
            del init_kwargs["task_type"]
        if "force_cache_reset" in init_kwargs:
            del init_kwargs["force_cache_reset"]

        return super().make_sampler_fn(
            task_type=HomeServiceTaskType.SIMPLE_PICK_AND_PLACE,
            force_cache_reset=True,
            **init_kwargs
        )
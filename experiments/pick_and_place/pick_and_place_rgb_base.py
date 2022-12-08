

from typing import Dict, Optional, Sequence

from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact.embodiedai.sensors.vision_sensors import DepthSensor
from env.tasks import HomeServiceTaskSampler
from experiments.home_service_base import HomeServiceBaseExperimentConfig


class PickAndPlaceRGBBaseExperimentConfig(HomeServiceBaseExperimentConfig):

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
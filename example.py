import os
import importlib
import inspect
import stringcase

from experiments.home_service_base import HomeServiceBaseExperimentConfig
from env.sensors import DepthHomeServiceSensor, RGBHomeServiceSensor
from env.tasks import HomeServiceTaskSampler, HomeServiceTaskSpecIterable
from env.environment import HomeServiceTaskSpec, HomeServiceMode
from env.constants import (
    STARTER_HOME_SERVICE_DATA_DIR,
    STEP_SIZE,
    VISIBILITY_DISTANCE,
    ROTATION_ANGLE,
    HORIZON_ANGLE,
    SMOOTHING_FACTOR,
    PROCTHOR_COMMIT_ID,
    PICKUPABLE_OBJECTS,
    OPENABLE_OBJECTS,
    RECEPTACLE_OBJECTS,
)
from allenact.base_abstractions.sensor import SensorSuite
from allenact.base_abstractions.experiment_config import ExperimentConfig
from allenact_plugins.clip_plugin.clip_preprocessors import (
    ClipResNetPreprocessor,
)
from ai2thor.platform import CloudRendering


# data = HomeServiceTaskSampler.load_home_service_data_from_path(
#     stage="debug",
#     base_dir=STARTER_HOME_SERVICE_DATA_DIR,
# )
# task_spec_iterator = HomeServiceTaskSpecIterable(
#     tasks_to_houses_to_task_spec_dicts=data,
#     seed=0,
#     epochs=1,
#     shuffle=False
# )

# task_spec: HomeServiceTaskSpec = next(task_spec_iterator)
# print(f'task_spec.unique_id: {task_spec.unique_id}')
# print(f'task_spec_iterator.current_task: {task_spec_iterator.current_task}')
# print(f'task_spec_iterator.current_house: {task_spec_iterator.current_house}')
# print(f'task_spec_iterator.remaining_tasks: {task_spec_iterator.remaining_tasks}')
# print(f'task_spec_iterator.remaining_houses: {task_spec_iterator.remaining_houses}')
# print(f'task_spec_iterator.houses_to_task_spec_dicts_for_current_task.keys(): {task_spec_iterator.houses_to_task_spec_dicts_for_current_task.keys()}')
# print(f'len(task_spec_iterator.task_spec_dicts_for_current_task_house): {len(task_spec_iterator.task_spec_dicts_for_current_task_house)}')

task_sampler_params = HomeServiceBaseExperimentConfig.stagewise_task_sampler_args(
    stage="debug",
    process_ind=0,
    total_processes=1,
    devices=[0, ],
)

task_sampler: HomeServiceTaskSampler = HomeServiceBaseExperimentConfig.make_sampler_fn(
    **task_sampler_params,
    force_cache_reset=True,
    repeats_per_task=3,
    epochs=2,
)

# QUALITY = "Very Low"
# SCREEN_SIZE = 224
# REFERENCE_SEGMENTATION = False
# ENV_KWARGS = dict(mode=HomeServiceMode.SNAP,)
# THOR_CONTROLLER_KWARGS = {
#     "gridSize": STEP_SIZE,
#     "visibilityDistance": VISIBILITY_DISTANCE,
#     "rotateStepDegrees": ROTATION_ANGLE,
#     "horizonStepDegrees": HORIZON_ANGLE,
#     "smoothingFactor": SMOOTHING_FACTOR,
#     "snapToGrid": True,
#     "quality": QUALITY,
#     "width": SCREEN_SIZE,
#     "height": SCREEN_SIZE,
#     "commit_id": PROCTHOR_COMMIT_ID,
#     "fastActionEmit": True,
#     "scene": "Procedural",
#     "renderDepthImage": True,
#     "renderSemanticSegmentation": REFERENCE_SEGMENTATION,
#     "renderInstanceSegmentation": REFERENCE_SEGMENTATION,
#     "gpu_device": 0,
#     "platform": CloudRendering,
# }

# PICKUP_ACTIONS = list(
#     sorted(
#         [
#             f"pickup_{stringcase.snakecase(object_type)}"
#             for object_type in PICKUPABLE_OBJECTS
#         ]
#     )
# )
# OPEN_ACTIONS = list(
#     sorted(
#         [
#             f"open_by_type_{stringcase.snakecase(object_type)}"
#             for object_type in OPENABLE_OBJECTS
#         ]
#     )
# )
# CLOSE_ACTIONS = list(
#     sorted(
#         [
#             f"close_by_type_{stringcase.snakecase(object_type)}"
#             for object_type in OPENABLE_OBJECTS
#         ]
#     )
# )
# PUT_ACTIONS = list(
#     sorted(
#         [
#             f"put_{stringcase.snakecase(object_type)}"
#             for object_type in RECEPTACLE_OBJECTS
#         ]
#     )
# )
# actions = (
#     ("done",)
#     + ("move_ahead",)
#     + ("move_left", "move_right", "move_back",)
#     + (
#         "rotate_right",
#         "rotate_left",
#         "stand",
#         "crouch",
#         "look_up",
#         "look_down",
#         *PICKUP_ACTIONS,
#         *OPEN_ACTIONS,
#         *CLOSE_ACTIONS,
#         *PUT_ACTIONS,
#     )
# )

# sensors = [
#     RGBHomeServiceSensor(
#         height=SCREEN_SIZE,
#         width=SCREEN_SIZE,
#         use_resnet_normalization=True,
#         uuid="rgb",
#         mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
#         stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
#     ),
#     DepthHomeServiceSensor(
#         height=SCREEN_SIZE,
#         width=SCREEN_SIZE,
#         use_normalization=False,
#         uuid="depth",
#     ),
# ]

# task_sampler: HomeServiceTaskSampler = HomeServiceTaskSampler.from_fixed_dataset(
#     stage="debug",
#     allowed_tasks=None,
#     allowed_pickup_objs=None,
#     allowed_target_receps=None,
#     home_service_env_kwargs=dict(
#         force_cache_reset=True,
#         **ENV_KWARGS,
#         controller_kwargs={
#             "x_display": None,
#             **THOR_CONTROLLER_KWARGS,
#         },
#     ),
#     seed=123,
#     sensors=SensorSuite(sensors),
#     max_steps=500,
#     discrete_actions=actions,
#     smooth_nav=False,
#     smoothing_factor=1,
#     require_done_action=True,
#     force_axis_aligned_start=True,
#     epochs=1,
#     expert_exploration_enabled=True,
#     repeats_per_task=3,
# )

print(f"Task Length: {task_sampler.length}")
print(f"Task Total Unique: {task_sampler.total_unique}")

task = task_sampler.next_task()
print(f"current repeat count: {task_sampler.cur_repeat_count}")
print(f"current_task_spec: {task.env.current_task_spec.unique_id}")

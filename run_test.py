import time
from tqdm import tqdm

from env.tasks import HomeServiceTaskSampler, HomeServiceTaskType
from experiments.home_service_base import HomeServiceBaseExperimentConfig

room_ind_to_type = {
    0: "Kitchen",
    1: "Living Room",
    2: "Bedroom",
    3: "Bathroom",
}

task_sampler_params = HomeServiceBaseExperimentConfig.stagewise_task_sampler_args(
    stage="train_unseen", process_ind=0, total_processes=1, headless=False, allowed_scene_inds=range(29, 31)
)
task_sampler: HomeServiceTaskSampler = HomeServiceBaseExperimentConfig.make_sampler_fn(
    **task_sampler_params,
    force_cache_reset=True,
    task_type=HomeServiceTaskType.SIMPLE_PICK_AND_PLACE,
    epochs=1,
)

num_tasks = task_sampler.total_unique
print(f'total unique: {num_tasks}')
success = 0

for i in tqdm(range(num_tasks)):
    
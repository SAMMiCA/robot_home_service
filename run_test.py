import time
from typing import cast
from allenact.algorithms.onpolicy_sync.policy import ActorCriticModel
from tqdm import tqdm
import torch

from env.tasks import HomeServiceTaskSampler, HomeServiceTaskType
from experiments.home_service_base import HomeServiceBaseExperimentConfig
from experiments.pick_and_place.pick_and_place_rgb_resnet_dagger import PickAndPlaceRGBResNetDaggerExperimentConfig

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

device = torch.deviec("cuda")
machine_params = PickAndPlaceRGBResNetDaggerExperimentConfig.machine_params("test")
sensor_preprocessor_graph = machine_params.sensor_preprocessor_graph.to(device)
create_model_kwargs = {"sensor_preprocessor_graph": sensor_preprocessor_graph}
net = cast(ActorCriticModel, PickAndPlaceRGBResNetDaggerExperimentConfig.create_model(**create_model_kwargs)).to(device)


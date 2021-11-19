import random
from typing import List
import torch
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from PIL import Image

from env.tasks import HomeServiceTaskSampler, HomeServiceTaskType
from experiments.home_service_base import HomeServiceBaseExperimentConfig
from room_change.utils import set_model
from room_change.main import parse_args as rc_parse_args

task_sampler_params = HomeServiceBaseExperimentConfig.stagewise_task_sampler_args(
    stage="train_seen", process_ind=0, total_processes=1, headless=False
)
task_sampler: HomeServiceTaskSampler = HomeServiceBaseExperimentConfig.make_sampler_fn(
    **task_sampler_params,
    force_cache_reset=True,
    task_type=HomeServiceTaskType.SIMPLE_PICK_AND_PLACE,
    epochs=1,
)
task = task_sampler.next_task()

device = torch.device('cuda')

# rc_opt = rc_parse_args()
# rc_opt.pretrained = './room_change/save/models/202110042117-scale30-noaugmentation/best_model.pth'
# rc_model = set_model(rc_opt, device)
# rc_model.eval()
# rc_transform = Compose(
#     [
#         Resize(224),
#         ToTensor(),
#         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]
# )

for i in range(2):
    print(f'{i}-th task')
    # task = task_sampler.next_task(pickup_target="Fork", place_target="DiningTable")
    task = task_sampler.next_task()
    while not task.is_done():
        obs = task.get_observations()
        action_ind, _ = task.query_expert()
        step_result = task.step(action=action_ind)
        
    import pdb; pdb.set_trace()

task_sampler.close()

print(f'finishied {num_tasks} tasks')
print(f'Success {success} out of {num_tasks}')


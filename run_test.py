import time
import random
import json
from datetime import timedelta
from typing import cast
from tqdm import tqdm
import torch

from allenact.algorithms.onpolicy_sync.policy import ActorCriticModel
from allenact.algorithms.onpolicy_sync.storage import RolloutStorage
from allenact.utils.tensor_utils import batch_observations

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
    stage="train_unseen", process_ind=0, total_processes=1, headless=False, 
    allowed_scene_inds=range(29, 31)
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
result_dict = {
    "num_tasks": num_tasks,
    "task_keys": [],
    "task_successes": [],
}

device = torch.device("cuda")
machine_params = PickAndPlaceRGBResNetDaggerExperimentConfig.machine_params("test")
sensor_preprocessor_graph = machine_params.sensor_preprocessor_graph.to(device)
create_model_kwargs = {"sensor_preprocessor_graph": sensor_preprocessor_graph}
net = cast(ActorCriticModel, PickAndPlaceRGBResNetDaggerExperimentConfig.create_model(**create_model_kwargs)).to(device)
net.load_state_dict(
    torch.load('./exp_PickAndPlaceRGBResNetDagger_8proc__stage_00__steps_000000250880.pt')['model_state_dict']
)

rollouts = RolloutStorage(num_steps=4, num_samplers=1, actor_critic=net)
rollouts.to(device)

st = time.time()
for i in range(num_tasks):
    print(f'{i+1}-th task')
    task_success = False
    task = task_sampler.next_task()
    result_dict["task_keys"].append(task_sampler.current_task_spec.metrics["task_key"])
    obs = task.get_observations()

    while not task.is_done():
        batch = batch_observations([obs], device)
        rollouts.insert_observations(
            sensor_preprocessor_graph.get_observations(batch)
        )
        with torch.no_grad():
            step_obs = rollouts.pick_observation_step(rollouts.step)
            memory = rollouts.pick_memory_step(rollouts.step)
            prev_actions = rollouts.pick_prev_actions_step(rollouts.step)
            net_output, memory = net(step_obs, memory, prev_actions, rollouts.masks[rollouts.step:rollouts.step+1])
        
            distr = net_output.distributions
            actions = distr.mode()

        action_net = actions.item()
        action_expert, _ = task.query_expert()

        if random.random() < 0.25:
            action_ind = action_expert
        else:
            action_ind = action_net

        step_result = task.step(action=action_ind)

    if step_result.info["action_name"] == "done":
        # print(f"{i+1}-th task success")
        task_success = True
        success += 1
    
    result_dict["task_successes"].append(task_success)
    if (i + 1) % 10 == 0:
        print(f"  > Success Rate: {float(success / (i + 1))} [ {success}/{i + 1}]")

task_sampler.close()
et = time.time()

result_dict["num_successes"] = success
result_dict["total_success_rate"] = round(float(success / num_tasks), 2)
result_dict["time_elapsed"] = str(timedelta(seconds=(et - st)))

print(f'finishied {num_tasks} tasks')
print(f'Total Success Rate: {result_dict["total_success_rate"]} [ {success}/{num_tasks} ]')
print(f'Time consumed: {result_dict["time_elapsed"]}')

with open('result.json', 'w') as f:
    json.dump(result_dict, f, indent=4)
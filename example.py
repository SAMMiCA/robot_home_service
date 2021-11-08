from collections import OrderedDict, defaultdict
import os

import numpy as np
from env.tasks import HomeServiceTaskSampler, HomeServiceTaskType
from experiments.home_service_base import HomeServiceBaseExperimentConfig
from PIL import Image
import argparse
import torch
import torch.backends.cudnn as cudnn
import pickle

task_sampler_params = HomeServiceBaseExperimentConfig.stagewise_task_sampler_args(
    stage="train", process_ind=0, total_processes=1, headless=False
)
task_sampler: HomeServiceTaskSampler = HomeServiceBaseExperimentConfig.make_sampler_fn(
    **task_sampler_params,
    task_type=HomeServiceTaskType.SIMPLE_PICK_AND_PLACE,
    force_cache_reset=True,
    epochs=1,
)

num_tasks = 500
success = 0

for i in range(num_tasks):
    print(f'{i}-th task')
    task = task_sampler.next_task()
    print(f'  task: {task.env.current_task_spec.task_type}')
    # print(f'  pickup_target: {task.env.current_task_spec.pickup_target}')
    # print(f'  place_target: {task.env.current_task_spec.place_target}')

    # rgb = task.env.last_event.frame
    # depth = task.env.last_event.depth_frame

    while not task.is_done():
        obs = task.get_observations()
        import pdb; pdb.set_trace()
        action_ind = int(input(f"action_ind="))

        # if task.num_steps_taken() % 10 == 0:
        #     print(
        #         f'step: {task.num_steps_taken()}:'
        #         f' taking action {task.action_names()[action_ind]}'
        #     )
        step_result = task.step(action=action_ind)
        # task.greedy_expert.update(
        #     action_taken=action_ind,
        #     action_success=step_result.info['action_success']
        # )

        if step_result.info['action_name'] == "done":
            success += 1
        
        # if task.current_subtask[0] == "Done":
        #     print(f"All subtasks DONE")
        #     import pdb; pdb.set_trace()
            
    print(f'{i}-th task done')

task_sampler.close()

print(f'finishied {num_tasks} tasks')
print(f'Success {success} out of {num_tasks}')


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
    stage="train", process_ind=0, total_processes=1, headless=False
)
task_sampler: HomeServiceTaskSampler = HomeServiceBaseExperimentConfig.make_sampler_fn(
    **task_sampler_params,
    task_type=HomeServiceTaskType.SIMPLE_PICK_AND_PLACE,
    force_cache_reset=True,
    epochs=1,
)

device = torch.device('cuda')

num_tasks = 500
success = 0
home_dicts = {f"Home{k}": [f"FloorPlan{k + i * 100}" for i in [0, 2, 3, 4]] for k in range(1, 31)}
room_ind_to_type = {
    0: "Kitchen",
    1: "Living Room",
    2: "Bedroom",
    3: "Bathroom",
}
room_ind_to_label = {
    0: "Kitchen",
    2: "Living Room",
    3: "Bedroom",
    4: "Bathroom",
}
command_to_action_str = {
    1: "MoveAhead",
    2: "RotateRight",
    3: "RotateLeft",
    4: "ResetScene"
}

rc_opt = rc_parse_args()
rc_opt.pretrained = './room_change/save/models/202110042117-scale30-noaugmentation/best_model.pth'
rc_model = set_model(rc_opt, device)
rc_model.eval()
rc_transform = Compose(
    [
        Resize(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

for i in range(num_tasks):
    print(f'{i}-th task')
    task = task_sampler.next_task(pickup_target="Fork", place_target="DiningTable")
    home_number = int(task.env.scene[9:]) % 100
    task_home = home_dicts[f"Home{home_number}"]
    
    scene_b = random.choice(task_home)
    task.env.controller.reset(scene=scene_b, width=512, height=512, quality="Ultra")
    print(f'scene: {scene_b}, label: {room_ind_to_label[int(scene_b[9:]) // 100]}')
    t_frames = None
    input_command = int(input("input_command: "))
    step = 0
    suffix = "b"
    detection = ""
    while input_command != -1:
        action_str = command_to_action_str[input_command]
        frame = task.env.controller.last_event.frame
        img = Image.fromarray(frame)
        img.save(f'./test/rc/{step}_{suffix}{detection}.png')
        t_frame = rc_transform(img)
        if t_frames is None:
            t_frames = t_frame.unsqueeze(0)
        else:
            if t_frames.shape[0] == 16:
                t_frames[:-1] = t_frames[1:].clone()
                t_frames[-1:] = t_frame.unsqueeze(0).clone()
                scene_b_pred = room_ind_to_type[rc_model(t_frames[:8].to(device))[1].sum(dim=0).argmax().cpu().item()]
                scene_a_pred = room_ind_to_type[rc_model(t_frames[8:].to(device))[1].sum(dim=0).argmax().cpu().item()]

                print(f'pred_b: {scene_b_pred}, pred_a: {scene_a_pred}')
                if scene_b_pred != scene_a_pred:
                    print("scene change detected")
                    detection = "_detected"
                else:
                    detection = ""

            elif t_frames.shape[0] < 16:
                t_frames = torch.cat([t_frames, t_frame.unsqueeze(0)], dim=0)
            else:
                raise RuntimeError()
        
        if input_command in [1, 2, 3]:
            task.env.controller.step(action_str)

        elif input_command == 4:
            scene_a = random.choice(task_home)
            task.env.controller.reset(scene=scene_a, width=512, height=512, quality="Ultra")
            print("scene changed")
            print(f'scene: {scene_a}, label: {room_ind_to_label[int(scene_a[9:]) // 100]}')
            if scene_a != scene_b:
                suffix = "a"

        step += 1
        input_command = int(input("input_command: "))


    # scene_b = random.choice(task_home)
    # task.env.controller.reset(scene=scene_b, width=512, height=512, quality="Ultra")

    # t_frames_b = None
    # for i in range(8):
    #     frame = task.env.controller.last_event.frame
    #     img = Image.fromarray(frame)
    #     img.save(f'./test/rc/b_{i}.png')
    #     t_frame = rc_transform(img)
    #     if t_frames_b is None:
    #         t_frames_b = t_frame.unsqueeze(0)
    #     else:
    #         t_frames_b = torch.cat([t_frames_b, t_frame.unsqueeze(0)], dim=0)
    #     # task.env.controller.step("RotateRight", degrees=45)
    #     input_command = int(input("input_command: "))
    #     action_str = command_to_action_str[input_command]
    #     # task.env.controller.step(action_str, degrees=45)
    #     task.env.controller.step(action_str)
    
    # scene_b_pred = room_ind_to_type[rc_model(t_frames_b.to(device))[1].sum(dim=0).argmax().cpu().item()]

    # scene_a = random.choice(task_home)
    # task.env.controller.reset(scene=scene_a, width=512, height=512, quality="Ultra")

    # t_frames_a = None
    # for i in range(8):
    #     frame = task.env.controller.last_event.frame
    #     img = Image.fromarray(frame)
    #     img.save(f'./test/rc/a_{i}.png')
    #     t_frame = rc_transform(img)
    #     if t_frames_a is None:
    #         t_frames_a = t_frame.unsqueeze(0)
    #     else:
    #         t_frames_a = torch.cat([t_frames_a, t_frame.unsqueeze(0)], dim=0)
    #     # task.env.controller.step("RotateRight", degrees=45)
    #     input_command = int(input("input_command: "))
    #     action_str = command_to_action_str[input_command]
    #     # task.env.controller.step(action_str, degrees=45)
    #     task.env.controller.step(action_str)
    
    # scene_a_pred = room_ind_to_type[rc_model(t_frames_a.to(device))[1].sum(dim=0).argmax().cpu().item()]
    # print(f'[before] scene: {scene_b}, pred: {scene_b_pred}, label: {room_ind_to_label[int(scene_b[9:]) // 100]}')
    # print(f'[after] scene: {scene_a}, pred: {scene_a_pred}, label: {room_ind_to_label[int(scene_a[9:]) // 100]}')



    import pdb; pdb.set_trace()
    print(f'  task: {task.env.current_task_spec.task_type}')

    while not task.is_done():
        obs = task.get_observations()
        # action_ind = int(input(f"action_ind="))

        # import pdb; pdb.set_trace()
        # action_ind, _ = task.query_expert()

        step_result = task.step(action=action_ind)

        if step_result.info['action_name'] == "done":
            success += 1
        
            
    print(f'{i}-th task done')

task_sampler.close()

print(f'finishied {num_tasks} tasks')
print(f'Success {success} out of {num_tasks}')


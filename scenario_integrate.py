import random
from tqdm import tqdm
import time
from datetime import timedelta
import os
import sys
import pickle
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from PIL import Image

from allenact.utils.system import get_logger
from env.tasks import HomeServiceTaskSampler, HomeServiceTaskType
from experiments.home_service_base import HomeServiceBaseExperimentConfig
from room_change.utils import set_model
from room_change.main import parse_args as rc_parse_args

import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from scene_graph.utils.functions import SavePath
from scene_graph.data import cfg, set_cfg, set_dataset, Config
from scene_graph.yolact import Yolact
from scene_graph.fusion_whole import rigid_transform
from scene_graph.utils.util import str2bool, calc_map, get_parser, get_view_frustum, get_camera_pose
from scene_graph.utils.augmentations import FastBaseTransform
from scene_graph.test_scan import Scan
from scene_graph.test_visualize import ScannetVis
from allenact.embodiedai.mapping.mapping_utils.point_cloud_utils import depth_frame_to_camera_space_xyz

# python scenario2.py --trained_model=scene_graph/weights/yolact_base_357_200000.pth --score_threshold=0.7 --dataset=robot_home_service_dataset
room_ind_to_type = {
    0: "Kitchen",
    1: "Living Room",
    2: "Bedroom",
    3: "Bathroom",
}

use_yolact = True
args = get_parser(use_yolact=use_yolact)
root_path = '/home/yhkim/python_projects/robot_home_service/robot_home_service/scene_graph'

if args.config is not None:
    set_cfg(args.config)
else:
    model_path = SavePath.from_str(args.trained_model)
    args.config = model_path.model_name + '_config'
    print('Config not specified. Parsed %s from the file name.\n' % args.config)
    set_cfg(args.config)

if args.detect:
    cfg.eval_mask_branch = False

if args.dataset is not None:
    set_dataset(args.dataset)
    cfg.num_classes = len(cfg.dataset.class_names) + 1

with torch.no_grad():
    if not os.path.exists('results'):
        os.makedirs('results')

    if args.cuda:
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if args.resume and not args.display:
        with open(args.ap_data_file, 'rb') as f:
            ap_data = pickle.load(f)
        calc_map(ap_data)
        exit()

    # dataset = None

    print('Loading model...', end='')
    net = Yolact()
    net.load_weights(args.trained_model)
    net.eval()
    print(' Done.')

    if args.cuda:
        net = net.cuda()

    net.detect.use_fast_nms = args.fast_nms
    net.detect.use_cross_class_nms = args.cross_class_nms
    cfg.mask_proto_debug = args.mask_proto_debug

mesh_plot = True
use_gpu = True
scannet_data = args.is_scannet
skip_imgs = 1

app = QApplication(sys.argv)

task_sampler_params = HomeServiceBaseExperimentConfig.stagewise_task_sampler_args(
    stage="train_unseen", process_ind=0, total_processes=1, headless=False, 
    # allowed_scene_inds=range(29, 31),
    # allowed_target_receps=["User"]
)
task_sampler: HomeServiceTaskSampler = HomeServiceBaseExperimentConfig.make_sampler_fn(
    **task_sampler_params,
    force_cache_reset=True,
    task_type=HomeServiceTaskType.SIMPLE_PICK_AND_PLACE,
    epochs=1,
)

device = torch.device('cuda')

args_to_remove = []
if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
        if arg[2:].split("=")[0] in args:
            args_to_remove.append(arg)
for arg in args_to_remove:
    sys.argv.remove(arg)

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

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.ion()
# plt.show()
num_tasks = task_sampler.total_unique
print(f'total {num_tasks} tasks')
success = 0

st = time.time()
for i in range(num_tasks):
    print(f'{i}-th task')
    task = task_sampler.next_task()
    obs = task.get_observations()
    t_frames = None
    init_scan = False
    while not task.is_done():
        subtask_type = obs['subtask']['type']
        manual = False
        if subtask_type == 1:
            # GOTO
            img = Image.fromarray((obs['rgb'] * 255).astype(np.uint8))
            t_frame = rc_transform(img)
            if t_frames is None:
                t_frames = t_frame.unsqueeze(0)
            elif t_frames.shape[0] == 8:
                pred_former = rc_model(t_frames[:4].to(device))[1].sum(dim=0).argmax().cpu().item()
                pred_latter = rc_model(t_frames[4:].to(device))[1].sum(dim=0).argmax().cpu().item()

                if pred_former != pred_latter:
                    print(f'Agent has moved to another room: {room_ind_to_type[pred_latter]}')
                    if obs['subtask']['target_type'] == pred_latter + 1:
                        print(f'Agent has arrived to the correct room to conduct the task')
            else:
                t_frames = torch.cat([t_frames, t_frame.unsqueeze(0)], dim=0)

        elif subtask_type == 2:
            # SCAN
            # manual = True
            # if not init_scan:
            #     scan = Scan(cam_intr=np.eye(3), mesh_plot=mesh_plot, scannet_data=scannet_data, mask_net=net, args=args, root_path=root_path, use_gpu=use_gpu)
            #     vis = ScannetVis(scan=scan, task=task, offset=0, skip_im=skip_imgs, mesh_plot=mesh_plot, parent=None)
            #     init_scan = True
            # else:
            #     vis.update_scan()
            # action_ind = random.randint(1, 7)
            # import pdb; pdb.set_trace()
            # action_ind = int(input('test: '))
            # Update SCAN Result from metadata
            objs = task.env.last_event.metadata['objects']
            target_obj = next(
                (
                    obj for obj in objs
                    if obj['objectType'] == task.env.current_task_spec.pickup_object
                ), None
            )
            # assert target_obj is not None
            if target_obj is None:
                break
            
            place_receptacles = [
                obj for obj in objs
                if obj['objectType'] == task.env.current_task_spec.place_receptacle
            ]
            if len(place_receptacles) == 0 and task.env.current_task_spec.place_receptacle != "User":
                break

            task.target_positions = {
                task.env.current_task_spec.pickup_object: target_obj["axisAlignedBoundingBox"]["center"],
                # task.env.current_task_spec.place_receptacle: place_receptacles[0]["axisAlignedBoundingBox"]["center"]
            }
            if task.env.current_task_spec.place_receptacle != "User":
                task.target_positions[task.env.current_task_spec.place_receptacle] = place_receptacles[0]["axisAlignedBoundingBox"]["center"]

            task._subtask_step += 1
            obs = task.get_observations()
            continue

        # ax.imshow((obs['rgb'] * 255).astype(np.uint8))
        # plt.draw()
        # plt.pause(0.01)
        # vis.update_scan()
        if not manual:
            action_ind, _ = task.query_expert()
        step_result = task.step(action=action_ind)
        obs = step_result.observation
    
    if step_result.info['action_name'] == "done":
        print(f"{i}-th task success")
        success += 1
    else:
        print(f'{i}-th task failed...')
        print(f'task: {task_sampler.current_task_spec.metrics}')

task_sampler.close()
et = time.time()

print(f'finishied {num_tasks} tasks')
print(f'Success {success} out of {num_tasks}')
print(f'Time consumed: {timedelta(seconds=(et - st))}')


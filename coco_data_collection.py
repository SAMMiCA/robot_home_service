import random
import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image 
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from skimage import measure
from itertools import groupby
from env.constants import OBJECT_TYPES_WITH_PROPERTIES, REARRANGE_SIM_OBJECTS, PICKUPABLE_OBJECTS, RECEPTACLE_OBJECTS, DEFAULT_COMPATIBLE_RECEPTACLES, SCENE_TO_SCENE_TYPE
from env.tasks import HomeServiceTaskSampler, HomeServiceTaskType
from env.sensors import InstanceSegmentationSensor, SubtaskHomeServiceSensor
from experiments.home_service_base import HomeServiceBaseExperimentConfig

HOME_SERVICE_OBJECTS = list(sorted(PICKUPABLE_OBJECTS + RECEPTACLE_OBJECTS))
INCLUDE_OTHER_MOVE_ACTIONS = False
OTHER_MOVE_ACTIONS = (tuple() if not INCLUDE_OTHER_MOVE_ACTIONS else ('move_left', 'move_right', 'move_back',))
ACTIONS = (('done', 'move_ahead') + OTHER_MOVE_ACTIONS + ('rotate_right', 'rotate_left', 'stand', 'crouch',
                                                          'look_up', 'look_down'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def close_contour(contour):
    # https://github.com/waspinator/pycococreator/blob/master/pycococreatortools/pycococreatortools.py
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_polygon(binary_mask, tolerance=0):
    # https://github.com/waspinator/pycococreator/blob/master/pycococreatortools/pycococreatortools.py
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:

        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation 
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


if __name__ == "__main__":
    num_iter_per_task = 150
    img_size = 224
    num_steps = 0
    img_idx = 0
    coco_id = 0


    # fig = plt.figure()
    # ax = []
    # ax.append(fig.add_subplot(2, 2, 1))
    # ax.append(fig.add_subplot(2, 2, 2))
    # ax.append(fig.add_subplot(2, 2, 3))
    # ax.append(fig.add_subplot(2, 2, 4))
    # plt.ion()
    # plt.show()

    # (seen_objs[train, val, test], unseen_objs[train, val, test])
    # ([train: seen_scenes, val: seen_scenes, test: unseen_scenes])
    annotations = [[[], [], []], [[], [], []]]
    images = [[[], [], []], [[], [], []]]
    ids_set = set()

    ROOT_PATH = 'data_detector'
    categories = ['seen_objs', 'unseen_objs']
    splits = ['train', 'val', 'test']
    image_paths = [os.path.join(ROOT_PATH, category, split) for split in splits for category in categories]
    for img_path in image_paths:
        if not os.path.exists(img_path):
            os.makedirs(img_path)

    thor_controller_kwargs = dict(
        renderSemanticSegmentation=True,
        renderInstanceSegmentation=True,
        quality="Ultra",
    )
    stages = ['train_seen', 'train_unseen', 'test_seen', 'test_unseen']
    for stage in stages:
        task_sampler_params = HomeServiceBaseExperimentConfig.stagewise_task_sampler_args(
            stage=stage, process_ind=0, total_processes=1, headless=False,
        )

        for sensor in task_sampler_params['sensors']:
            if isinstance(sensor, SubtaskHomeServiceSensor):
                task_sampler_params['sensors'].remove(sensor)

        task_sampler_params['sensors'].append(
            InstanceSegmentationSensor(ordered_object_types=HOME_SERVICE_OBJECTS)
        )
        task_sampler: HomeServiceTaskSampler = HomeServiceBaseExperimentConfig.make_sampler_fn(
            **task_sampler_params,
            task_type=HomeServiceTaskType.REARRANGE,
            thor_controller_kwargs=thor_controller_kwargs,
            force_cache_reset=True,
            runtime_sample=False,
            repeats_before_scene_change=1,
            epochs=1,
        )

        task = task_sampler.next_task()
        task.env.reset(
            task_spec=task_sampler.current_task_spec,
            scene_type=SCENE_TO_SCENE_TYPE[task_sampler.current_task_spec.target_scene]
        )
        obs = task.get_observations()
        for iter in range(task_sampler.total_unique):
            print(f'Collection progress... [{iter + 1}/{task_sampler.total_unique}]')
            category_obj = task_sampler.stage.split('_')[0]
            if category_obj == "train":
                category_obj_idx = 0
            elif category_obj == "test":
                category_obj_idx = 1
            
            category_scene = task_sampler.stage.split('_')[1]
            if category_scene == "seen":
                if task_sampler.current_task_spec.scene_index % 10 != 0:
                    category_scene_idx = 0
                else:
                    category_scene_idx = 1
            elif category_scene == "unseen":
                category_scene_idx = 2

            for poses in task_sampler.current_task_spec.starting_poses:
                obj_name = poses['name']
                obj = next(
                    (
                        obj 
                        for obj in task.env.last_event.metadata['objects']
                        if obj['name'] == obj_name
                    ), None
                )
                if obj is None:
                    continue

                tp_success = False
                interactable_positions = task.env._interactable_positions_cache.get(
                    scene_name=task.env.scene, obj=obj, controller=task.env.controller
                )
                random.shuffle(interactable_positions)
                while not tp_success and len(interactable_positions) > 0: 
                    tp_position = interactable_positions.pop()
                    tp_pos = dict(x=tp_position["x"], y=tp_position["y"], z=tp_position["z"])
                    tp_rot = dict(x=0, y=tp_position["rotation"], z=0)
                    tp_std = tp_position["standing"]
                    tp_hor = tp_position["horizon"]
                    task.env.controller.step(
                        action="TeleportFull", position=tp_pos, rotation=tp_rot, horizon=tp_hor, standing=tp_std
                    )
                    tp_success = task.env.last_event.metadata["lastActionSuccess"]
                
                obs = task.get_observations()
                add_img = False
                for i in range(obs["instance_segmentation"]["inst_label"].shape[0]):
                    x_min, y_min, x_max, y_max = obs['instance_segmentation']['inst_bbox'][i]
                    x_min = int(x_min)
                    y_min = int(y_min)
                    x_max = int(x_max)
                    y_max = int(y_max)
                    area = (x_max - x_min) * (y_max - y_min)
                    if (x_max - x_min) < 15 or (y_max - y_min) < 15:
                        continue
                    mask = obs['instance_segmentation']['inst_mask'][i]
                    poly = binary_mask_to_polygon(mask)
                    data_anno = dict(
                        image_id=img_idx,
                        id=coco_id,
                        category_id=int(obs['instance_segmentation']['inst_label'][i]),
                        bbox=[x_min, y_min, x_max-x_min, y_max-y_min],
                        area=area,
                        segmentation=poly,
                        iscrowd=0
                    )
                    annotations[category_obj_idx][category_scene_idx].extend([data_anno])
                    coco_id += 1
                    add_img = True

                if add_img:
                    img_rgb = Image.fromarray((obs['rgb'] * 255).astype(np.uint8))
                    img_name = os.path.join(image_paths[category_obj_idx][category_scene_idx], f'{img_idx:06d}.png')
                    img_rgb.save(img_name)
                    images[category_obj_idx][category_scene_idx].append(
                        dict(
                            id=img_idx,
                            file_name=img_name,
                            height=img_size,
                            width=img_size
                        )
                    )
                    img_idx += 1

            task = task_sampler.next_task()

        task_sampler.close()

    for category_object_idx, category_object in enumerate(categories):
        for category_scene_idx, category_scene in enumerate(splits):
            coco_format_json = dict(
                images=images[category_obj_idx][category_scene_idx],
                annotations=annotations[category_obj_idx][category_scene_idx],
                categories=[{'id': i, 'name': t} for i, t in enumerate(HOME_SERVICE_OBJECTS)]
            )
            with open(os.path.join(ROOT_PATH, category_object, f'anno_{category_scene}.json'), 'w') as f:
                json.dump(coco_format_json, f)


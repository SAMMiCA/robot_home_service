"""A script for generating robot home service datasets."""

import argparse
import json
import multiprocessing as mp
import os
import pickle
import queue
import random
import time
from collections import defaultdict
from typing import List, Sequence, Set, Dict, Optional, Any, cast

import compress_pickle
import numpy as np
import tqdm
from ai2thor.controller import Controller

from allenact.utils.misc_utils import md5_hash_str_as_int
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from datagen.datagen_constants import OBJECTS_FOR_TEST, TASK_ORDERS
from datagen.datagen_utils import (
    get_scenes,
    get_random_seeds,
    filter_pickupable,
    open_objs,
    get_object_ids_to_not_move_from_object_types,
    # remove_objects_until_all_have_identical_meshes,
    find_object_by_type,
    scene_from_type_idx
)
from env.constants import OBJECT_TYPES_WITH_PROPERTIES, SCENE_TYPE_TO_SCENES, STARTER_HOME_SERVICE_DATA_DIR, THOR_COMMIT_ID
from env.environment import (
    HomeServiceTHOREnvironment,
    HomeServiceTaskSpec,
)

mp = mp.get_context("spawn")


def generate_one_task_order_given_initial_conditions(
    controller: Controller,
    scene: str,
    start_kwargs: dict,
    target_kwargs: dict,
    pickup_object_type: str,
    start_receptacle_type: str,
    place_receptacle_type: str,
    # obj_rearrangement_count: int,
    # object_types_to_not_move: Set[str],
    agent_pos: Dict[str, float],
    agent_rot: Dict[str, float],
):
    # nonpickupable_open_count = random.randint(0, 1)
    # obj_rearrangement_count -= nonpickupable_open_count

    # Start position
    controller.reset(scene)
    controller.step(
        "TeleportFull", horizon=0, standing=True, rotation=agent_rot, **agent_pos
    )
    if not controller.last_event.metadata["lastActionSuccess"]:
        print("a")
        return None, None

    excluded_receptacles = [start_receptacle_type]
    if place_receptacle_type != "User":
        excluded_receptacles.append(place_receptacle_type)
    controller.step("InitialRandomSpawn", excludedReceptacles=excluded_receptacles, **start_kwargs)
    if not controller.last_event.metadata["lastActionSuccess"]:
        print("b")
        return None, None

    for _ in range(12):
        controller.step("Pass")

    if any(o["isBroken"] for o in controller.last_event.metadata["objects"]):
        print("c")
        return None, None

    # get initial and post random spawn object data
    objects_after_first_irs = controller.last_event.metadata["objects"]
    openable_objects = [
        obj
        for obj in objects_after_first_irs
        if obj["openable"] and not obj["pickupable"]
    ]
    random.shuffle(openable_objects)
        
    # accounts for possibly a rare event that I cannot think of, where opening
    # a non-pickupable object moves a pickupable object.
    pickupable_objects_after_first_irs = [
        obj
        for obj in objects_after_first_irs
        if obj["pickupable"]
    ]

    pickup_target = next(
        (
            obj 
            for obj in pickupable_objects_after_first_irs
            if obj["objectType"] == pickup_object_type
        ),
        None
    )
    assert pickup_target is not None

    controller.step(
        "TeleportFull", horizon=0, standing=True, rotation=agent_rot, **agent_pos
    )
    if not controller.last_event.metadata["lastActionSuccess"]:
        print("d")
        return None, None

    pickupable_objects_after_shuffle: Optional[List[Dict[str, Any]]] = None

    # choose one of the place receptacle object and open it
    start_receptacles = [
        obj
        for obj in objects_after_first_irs
        if obj["objectType"] == start_receptacle_type
    ]
    random.shuffle(start_receptacles)
    
    place_receptacles = [
        obj
        for obj in objects_after_first_irs
        if obj["objectType"] == place_receptacle_type
    ]
    random.shuffle(place_receptacles)

    objs_to_open = []
    place_success = False if place_receptacle_type != "User" else True
    for place_receptacle in place_receptacles:
        controller.step(
            "GetSpawnCoordinatesAboveReceptacle",
            objectId=place_receptacle["objectId"],
            anywhere=True,
        )
        if not controller.last_event.metadata["lastActionSuccess"]:
            continue
        
        coords_above_place_receptacle = controller.last_event.metadata["actionReturn"]
        random.shuffle(coords_above_place_receptacle)

        while len(coords_above_place_receptacle) > 0:
            controller.step(
                "PlaceObjectAtPoint",
                objectId=pickup_target["objectId"],
                position=coords_above_place_receptacle.pop(),
            )
            if not controller.last_event.metadata["lastActionSuccess"]:
                continue
            else:
                place_success = True
                break
        
        if place_success:
            if place_receptacle["openable"]:
                controller.step(
                    "OpenObject",
                    objectId=place_receptacle["objectId"],
                    openness=1,
                    forceAction=True,
                )
                objs_to_open.append(place_receptacle["name"])
            break
    
    # if not place_success:
    #     print('relocate objects in place_receptacles')
    #     for place_receptacle in place_receptacles:
    #         if place_receptacle["receptacleObjectIds"] is None:
    #             continue
    #         receptacle_objects = [
    #             obj
    #             for obj in objects_after_first_irs
    #             if obj["objectId"] in place_receptacle["receptacleObjectIds"]
    #         ]
    #         tt = [o['objectType'] for o in receptacle_objects]
    #         print(f'{tt} | {place_receptacle["objectType"]}')
    #         excluded_object_ids = [
    #             obj["objectId"]
    #             for obj in objects_after_first_irs
    #             if obj not in receptacle_objects
    #         ]
    #         controller.step(
    #             "InitialRandomSpawn",
    #             excludedObjectIds=excluded_object_ids,
    #             excludedReceptacles=[place_receptacle_type, start_receptacle_type],
    #             **target_kwargs
    #         )
    #         if not controller.last_event.metadata["lastActionSuccess"]:
    #             print('1')
    #             continue

    #         controller.step(
    #             "GetSpawnCoordinatesAboveReceptacle",
    #             objectId=place_receptacle["objectId"],
    #             anywhere=True,
    #         )
    #         if not controller.last_event.metadata["lastActionSuccess"]:
    #             print('2')
    #             continue
            
    #         coords_above_place_receptacle = controller.last_event.metadata["actionReturn"]
    #         random.shuffle(coords_above_place_receptacle)

    #         for retry_ind in range(10):
    #             controller.step(
    #                 "PlaceObjectAtPoint",
    #                 objectId=pickup_target["objectId"],
    #                 position=coords_above_place_receptacle.pop(),
    #             )
    #             if not controller.last_event.metadata["lastActionSuccess"]:
    #                 print('3')
    #                 print(controller.last_event)
    #                 continue
    #             else:
    #                 place_success = True
    #                 break
            
    #         if place_success:
    #             if place_receptacle["openable"]:
    #                 controller.step(
    #                     "OpenObject",
    #                     objectId=place_receptacle["objectId"],
    #                     openness=1,
    #                     forceAction=True,
    #                 )
    #                 objs_to_open.append(place_receptacle["name"])
    #             break
                
    start_success = False
    for start_receptacle in start_receptacles:
        controller.step(
            "GetSpawnCoordinatesAboveReceptacle",
            objectId=start_receptacle["objectId"],
            anywhere=True,
        )
        if not controller.last_event.metadata["lastActionSuccess"]:
            continue
        
        coords_above_start_receptacle = controller.last_event.metadata["actionReturn"]
        random.shuffle(coords_above_start_receptacle)

        while len(coords_above_start_receptacle) > 0:
            controller.step(
                "PlaceObjectAtPoint",
                objectId=pickup_target["objectId"],
                position=coords_above_start_receptacle.pop(),
            )
            if not controller.last_event.metadata["lastActionSuccess"]:
                continue
            else:
                start_success = True
                break
        
        if start_success:
            if start_receptacle["openable"]:
                controller.step(
                    "OpenObject",
                    objectId=start_receptacle["objectId"],
                    openness=1,
                    forceAction=True,
                )
                objs_to_open.append(start_receptacle["name"])
            for _ in range(12):
                # This shouldn't be necessary but we run these actions
                # to let physics settle.
                controller.step("Pass")
            pickupable_objects_after_shuffle = [
                obj
                for obj in controller.last_event.metadata["objects"]
                if obj['pickupable']
            ]
            break

    # if not start_success:
    #     for start_receptacle in start_receptacles:
    #         if start_receptacle["receptacleObjectIds"] is None:
    #             continue
    #         receptacle_objects = [
    #             obj
    #             for obj in objects_after_first_irs
    #             if obj["objectId"] in start_receptacle["receptacleObjectIds"]
    #         ]
    #         excluded_object_ids = [
    #             obj["objectId"]
    #             for obj in objects_after_first_irs
    #             if obj not in receptacle_objects
    #         ]
    #         excluded_receptacles = [start_receptacle_type]
    #         if place_receptacle_type != "User":
    #             excluded_receptacles.append(place_receptacle_type)
    #         controller.step(
    #             "InitialRandomSpawn",
    #             excludedObjectIds=excluded_object_ids,
    #             excludedReceptacles=excluded_receptacles,
    #             **target_kwargs
    #         )
    #         if not controller.last_event.metadata["lastActionSuccess"]:
    #             continue

    #         controller.step(
    #             "GetSpawnCoordinatesAboveReceptacle",
    #             objectId=start_receptacle["objectId"],
    #             anywhere=True,
    #         )
    #         if not controller.last_event.metadata["lastActionSuccess"]:
    #             continue
            
    #         coords_above_start_receptacle = controller.last_event.metadata["actionReturn"]
    #         random.shuffle(coords_above_start_receptacle)

    #         for retry_ind in range(10):
    #             controller.step(
    #                 "PlaceObjectAtPoint",
    #                 objectId=pickup_target["objectId"],
    #                 position=coords_above_start_receptacle.pop(),
    #             )
    #             if not controller.last_event.metadata["lastActionSuccess"]:
    #                 continue
    #             else:
    #                 start_success = True
    #                 break
            
    #         if start_success:
    #             if start_receptacle["openable"]:
    #                 controller.step(
    #                     "OpenObject",
    #                     objectId=start_receptacle["objectId"],
    #                     openness=1,
    #                     forceAction=True,
    #                 )
    #                 objs_to_open.append(start_receptacle["name"])
    #             break

    for o in controller.last_event.metadata["objects"]:
        if o["isBroken"]:
            print(
                f"In scene {controller.last_event.metadata['objects']},"
                f" object {o['name']} broke during setup."
            )
            return None, None

    if not start_success or not place_success:
        print(f'start_success: {start_success} | place_success: {place_success}')
        return None, None

    pickupable_objects_after_first_irs.sort(key=lambda x: x["name"])
    pickupable_objects_after_shuffle.sort(key=lambda x: x["name"])

    if any(
        o0["name"] != o1["name"]
        for o0, o1 in zip(
            pickupable_objects_after_first_irs, pickupable_objects_after_shuffle
        )
    ):
        print("Pickupable object names don't match after shuffle!")
        return None, None

    return (
        [
            {
                "name": pickupable_objects_after_shuffle[i]["name"],
                "objectName": pickupable_objects_after_shuffle[i]["name"],
                "position": pickupable_objects_after_shuffle[i]["position"],
                "rotation": pickupable_objects_after_shuffle[i]["rotation"],
            }
            for i in range(len(pickupable_objects_after_shuffle))
        ],
        [
            {
                "name": open_obj_name,
                "objectName": open_obj_name,
                "objectId": next(
                    o["objectId"]
                    for o in openable_objects
                    if o["name"] == open_obj_name
                ),
            }
            for open_obj_name in objs_to_open
        ]
    )


def generate_task_specs_for_task_orders(
    stage_seed: int,
    # stage_scenes: List[str],
    env: HomeServiceTHOREnvironment,
    task_orders: Sequence[Dict[str, Any]],
    # object_types_to_not_move: Set[str],
    # max_obj_rearrangements_per_scene: int = 5,
    # scene_reuse_count: int = 50,
    # obj_name_to_avoid_positions: Optional[Dict[str, np.ndarray]] = None,
    force_visible: bool = True,
    place_stationary: bool = False,
    rotation_increment: int = 30,
) -> dict:
    if 360 % rotation_increment != 0:
        raise ValueError("Rotation increment must be a factor of 360")

    # if obj_name_to_avoid_positions is None:
    #     obj_name_to_avoid_positions = defaultdict(
    #         lambda: np.array([[-1000, -1000, -1000]])
    #     )

    controller = env.controller

    out: dict = dict()
    for task_order in task_orders:
        start_scene_types = task_order["startSceneType"]
        random.shuffle(start_scene_types)
        start_scene_type = next(iter(start_scene_types), None)
        assert start_scene_type is not None
        target_scene_type = task_order["targetSceneType"]

        target_receptacle_type = task_order["targetReceptacleType"]
        start_receptacle_type = task_order["startReceptacleType"]
        pickup_object_types = task_order["pickupObjectTypes"]
        for pickup_object_type in pickup_object_types:
            # task_key = (
            #     pickup_object_type,
            #     start_receptacle_type,
            #     target_receptacle_type,
            # )
            task_key = (
                f'Pick_{pickup_object_type}_On_{start_receptacle_type}_And_Place_{target_receptacle_type}'
                if target_receptacle_type != "User" else
                f'Bring_Me_{pickup_object_type}_On_{start_receptacle_type}'
            )
            out[task_key] = []
            for i in range(1, 31):                   
                target_scene = scene_from_type_idx(target_scene_type, i)
                seed = md5_hash_str_as_int(f"{stage_seed}|{target_scene}")
                random.seed(seed)

                controller.reset(target_scene)
                pickup_objects = find_object_by_type(
                    controller.last_event.metadata["objects"],
                    pickup_object_type
                )

                # NO pickup_object in the target_scene
                if len(pickup_objects) == 0:
                    continue

                start_receptacles = find_object_by_type(
                    controller.last_event.metadata["objects"],
                    start_receptacle_type
                )

                if len(start_receptacles) == 0:
                    continue

                target_receptacles = find_object_by_type(
                    controller.last_event.metadata["objects"],
                    target_receptacle_type
                )

                if len(target_receptacles) == 0 and target_receptacle_type != "User":
                    continue
                
                agent_pos_rots = dict()
                for scene_type in SCENE_TYPE_TO_SCENES:
                    scene = scene_from_type_idx(scene_type, i)
                    seed = md5_hash_str_as_int(f"{stage_seed}|{scene}")
                    random.seed(seed)

                    controller.reset(scene)
                    evt = controller.step("GetReachablePositions")
                    rps: List[Dict[str, float]] = evt.metadata["actionReturn"]
                    rps.sort(key=lambda d: (round(d["x"], 2), round(d["z"], 2)))
                    rotations = np.arange(0, 360, rotation_increment)

                    pos = random.choice(rps)
                    rot = {"x": 0, "y": int(random.choice(rotations)), "z": 0}
                    agent_pos_rots[scene_type] = {
                        "position": pos,
                        "rotation": rot,
                    }
                
                # used to make sure the positions of the objects
                # are not always the same across the same scene.
                start_kwargs = {
                    "randomSeed": random.randint(0, int(1e7) - 1),
                    "forceVisible": force_visible,
                    "placeStationary": place_stationary,
                }
                target_kwargs = {
                    "randomSeed": random.randint(0, int(1e7) - 1),
                    "forceVisible": force_visible,
                    "placeStationary": place_stationary,
                }

                starting_poses, objs_to_open = generate_one_task_order_given_initial_conditions(
                    controller=controller,
                    scene=target_scene,
                    start_kwargs=start_kwargs,
                    target_kwargs=target_kwargs,
                    pickup_object_type=pickup_object_type,
                    start_receptacle_type=start_receptacle_type,
                    place_receptacle_type=target_receptacle_type,
                    agent_pos=agent_pos_rots[target_scene_type]["position"],
                    agent_rot=agent_pos_rots[target_scene_type]["rotation"],
                )
                if starting_poses is None:
                    print(
                        f"Skipping {target_scene}, {agent_pos_rots[target_scene_type]['position']}, {int(agent_pos_rots[target_scene_type]['rotation']['y'])} {pickup_object_type}, {start_receptacle_type}, {target_receptacle_type}."
                    )
                    continue
                
                task_spec_dict = {
                    "agent_positions": {
                        scene_type: agent_pos_rots[scene_type]['position']
                        for scene_type in SCENE_TYPE_TO_SCENES
                    },
                    "agent_rotations": {
                        scene_type: int(agent_pos_rots[scene_type]['rotation']['y'])
                        for scene_type in SCENE_TYPE_TO_SCENES
                    },
                    "scene_index": i,
                    "start_scene": scene_from_type_idx(start_scene_type, i),
                    "target_scene": target_scene,
                    "pickup_object": pickup_object_type,
                    "start_receptacle": start_receptacle_type,
                    "place_receptacle": target_receptacle_type,
                    "starting_poses": starting_poses,
                    "objs_to_open": objs_to_open,
                }
                env.reset(task_spec=HomeServiceTaskSpec(**task_spec_dict), scene_type=target_scene_type)

                reachable_positions = env.controller.step(
                    "GetReachablePositions"
                ).metadata["actionReturn"]

                # check whether pickup object is interactable
                interactable_poses = env.controller.step(
                    "GetInteractablePoses",
                    objectId=next(
                        o["objectId"]
                        for o in env.controller.last_event.metadata["objects"]
                        if o["objectType"] == pickup_object_type
                    ),
                    positions=reachable_positions,
                ).metadata["actionReturn"]
                if interactable_poses is None or len(interactable_poses) == 0:
                    continue
                
                # check whether place target is interactable
                if OBJECT_TYPES_WITH_PROPERTIES[target_receptacle_type]["openable"]:
                    obj = next(
                        obj
                        for obj in objs_to_open
                        if obj["name"].split("_")[0] == target_receptacle_type
                    )
                else:
                    obj = next(
                        obj
                        for obj in env.controller.last_event.metadata['objects']
                        if obj['objectType'] == target_receptacle_type
                    )
                interactable_poses = env.controller.step(
                    "GetInteractablePoses",
                    objectId=obj["objectId"],
                    positions=reachable_positions,
                ).metadata["actionReturn"]
                if interactable_poses is None or len(interactable_poses) == 0:
                    continue
                
                out[task_key].append(task_spec_dict)

        # for pickup_object_type in pickup_object_types:
        #     for i in range(1, 31):
        #         try_count = 0
        #         while True:
        #             try_count += 1
        #             if try_count > 100:
        #                 raise RuntimeError(
        #                     f"FFFFFF"
        #                 )
                    
        #             target_scene = scene_from_type_idx(target_scene_type, i)
        #             seed = md5_hash_str_as_int(f"{stage_seed}|{target_scene}")
        #             random.seed(seed)

        #             controller.reset(target_scene)
        #             pickup_objects = find_object_by_type(
        #                 controller.last_event.metadata["objects"],
        #                 pickup_object_type
        #             )

        #             # NO pickup_object in the target_scene
        #             if len(pickup_objects) == 0:
        #                 break

        #             start_receptacles = find_object_by_type(
        #                 controller.last_event.metadata["objects"],
        #                 start_receptacle_type
        #             )

        #             if len(start_receptacles) == 0:
        #                 break

        #             target_receptacles = find_object_by_type(
        #                 controller.last_event.metadata["objects"],
        #                 target_receptacle_type
        #             )

        #             if len(target_receptacles) == 0 and target_receptacle_type != "User":
        #                 break
                    
        #             agent_pos_rots = dict()
        #             for scene_type in SCENE_TYPE_TO_SCENES:
        #                 scene = scene_from_type_idx(scene_type, i)
        #                 seed = md5_hash_str_as_int(f"{stage_seed}|{scene}")
        #                 random.seed(seed)

        #                 controller.reset(scene)
        #                 evt = controller.step("GetReachablePositions")
        #                 rps: List[Dict[str, float]] = evt.metadata["actionReturn"]
        #                 rps.sort(key=lambda d: (round(d["x"], 2), round(d["z"], 2)))
        #                 rotations = np.arange(0, 360, rotation_increment)

        #                 pos = random.choice(rps)
        #                 rot = {"x": 0, "y": int(random.choice(rotations)), "z": 0}
        #                 agent_pos_rots[scene_type] = {
        #                     "position": pos,
        #                     "rotation": rot,
        #                 }
                    
        #             # used to make sure the positions of the objects
        #             # are not always the same across the same scene.
        #             start_kwargs = {
        #                 "randomSeed": random.randint(0, int(1e7) - 1),
        #                 "forceVisible": force_visible,
        #                 "placeStationary": place_stationary,
        #             }
        #             target_kwargs = {
        #                 "randomSeed": random.randint(0, int(1e7) - 1),
        #                 "forceVisible": force_visible,
        #                 "placeStationary": place_stationary,
        #             }

        #             starting_poses, objs_to_open = generate_one_task_order_given_initial_conditions(
        #                 controller=controller,
        #                 scene=target_scene,
        #                 start_kwargs=start_kwargs,
        #                 target_kwargs=target_kwargs,
        #                 pickup_object_type=pickup_object_type,
        #                 start_receptacle_type=start_receptacle_type,
        #                 place_receptacle_type=target_receptacle_type,
        #                 agent_pos=agent_pos_rots[target_scene_type]["position"],
        #                 agent_rot=agent_pos_rots[target_scene_type]["rotation"],
        #             )
        #             if starting_poses is None:
        #                 print(
        #                     f"Skipping {target_scene}, {agent_pos_rots[target_scene_type]['position']}, {int(agent_pos_rots[target_scene_type]['rotation']['y'])} {pickup_object_type}, {start_receptacle_type}, {target_receptacle_type}."
        #                 )
        #                 continue
                    
        #             task_spec_dict = {
        #                 "agent_positions": {
        #                     scene_type: agent_pos_rots[scene_type]['position']
        #                     for scene_type in SCENE_TYPE_TO_SCENES
        #                 },
        #                 "agent_rotations": {
        #                     scene_type: int(agent_pos_rots[scene_type]['rotation']['y'])
        #                     for scene_type in SCENE_TYPE_TO_SCENES
        #                 },
        #                 "scene_index": i,
        #                 "start_scene": scene_from_type_idx(start_scene_type, i),
        #                 "target_scene": target_scene,
        #                 "pickup_object": pickup_object_type,
        #                 "start_receptacle": start_receptacle_type,
        #                 "place_receptacle": target_receptacle_type,
        #                 "starting_poses": starting_poses,
        #                 "objs_to_open": objs_to_open,
        #             }

        #             env.reset(task_spec=HomeServiceTaskSpec(**task_spec_dict), scene_type=target_scene_type)

        #             reachable_positions = env.controller.step(
        #                 "GetReachablePositions"
        #             ).metadata["actionReturn"]

        #             # check whether pickup object is interactable
        #             interactable_poses = env.controller.step(
        #                 "GetInteractablePoses",
        #                 objectId=next(
        #                     o["objectId"]
        #                     for o in env.controller.last_event.metadata["objects"]
        #                     if o["objectType"] == pickup_object_type
        #                 ),
        #                 positions=reachable_positions,
        #             ).metadata["actionReturn"]
        #             if interactable_poses is None or len(interactable_poses) == 0:
        #                 continue
                    
        #             # check whether place target is interactable
        #             if OBJECT_TYPES_WITH_PROPERTIES[target_receptacle_type]["openable"]:
        #                 obj = next(
        #                     obj
        #                     for obj in objs_to_open
        #                     if obj["name"].split("_")[0] == target_receptacle_type
        #                 )
        #             else:
        #                 obj = next(
        #                     obj
        #                     for obj in env.controller.last_event.metadata['objects']
        #                     if obj['objectType'] == target_receptacle_type
        #                 )
        #             interactable_poses = env.controller.step(
        #                 "GetInteractablePoses",
        #                 objectId=obj["objectId"],
        #                 positions=reachable_positions,
        #             ).metadata["actionReturn"]
        #             if interactable_poses is None or len(interactable_poses) == 0:
        #                 continue
                    
        #             out[i].append(task_spec_dict)
        #             break
    
    return out

def rearrangement_datagen_worker(
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    # scene_to_obj_name_to_avoid_positions: Optional[
    #     Dict[str, Dict[str, np.ndarray]]
    # ] = None,
):
    env = HomeServiceTHOREnvironment(
        force_cache_reset=True, controller_kwargs={"commit_id": THOR_COMMIT_ID}
    )

    while True:
        try:
            single_task_order, stage, seed = input_queue.get(timeout=2)
            # key = (
            #     single_task_order["pickupObjectTypes"][0],
            #     single_task_order["startReceptacleType"],
            #     single_task_order["targetReceptacleType"],
            # )
            key = (
                f'Pick_{single_task_order["pickupObjectTypes"][0]}_On_{single_task_order["startReceptacleType"]}_And_Place_{single_task_order["targetReceptacleType"]}'
                if single_task_order["targetReceptacleType"] != "User" else
                f'Bring_Me_{single_task_order["pickupObjectTypes"][0]}_On_{single_task_order["startReceptacleType"]}'
            )
        except queue.Empty:
            break
        data = generate_task_specs_for_task_orders(
            stage_seed=seed,
            task_orders=[single_task_order],
            env=env,
            # object_types_to_not_move=OBJECT_TYPES_TO_NOT_MOVE,
            # obj_name_to_avoid_positions=None
            # if scene_to_obj_name_to_avoid_positions is None
            # else scene_to_obj_name_to_avoid_positions[scene],
        )
        output_queue.put((key, stage, data[key]))


# def get_scene_to_obj_name_to_seen_positions():
#     scene_to_task_spec_dicts = compress_pickle.load(
#         os.path.join(STARTER_DATA_DIR, f"train.pkl.gz")
#     )
#     assert len(scene_to_task_spec_dicts) == 80 and all(
#         len(v) == 50 for v in scene_to_task_spec_dicts.values()
#     )

#     scene_to_obj_name_to_positions = {}
#     for scene in tqdm.tqdm(scene_to_task_spec_dicts):
#         obj_name_to_positions = defaultdict(lambda: [])
#         for task_spec_dict in scene_to_task_spec_dicts[scene]:
#             for od in task_spec_dict["openable_data"]:
#                 obj_name_to_positions[od["name"]].extend(
#                     (od["start_openness"], od["target_openness"])
#                 )

#             for sp, tp in zip(
#                 task_spec_dict["starting_poses"], task_spec_dict["target_poses"]
#             ):
#                 assert sp["name"] == tp["name"]

#                 position_dist = IThorEnvironment.position_dist(
#                     sp["position"], tp["position"]
#                 )
#                 rotation_dist = IThorEnvironment.angle_between_rotations(
#                     sp["rotation"], tp["rotation"]
#                 )
#                 if position_dist >= 1e-2 or rotation_dist >= 5:
#                     obj_name_to_positions[sp["name"]].append(
#                         [sp["position"][k] for k in ["x", "y", "z"]]
#                     )
#                     obj_name_to_positions[sp["name"]].append(
#                         [tp["position"][k] for k in ["x", "y", "z"]]
#                     )
#         scene_to_obj_name_to_positions[scene] = {
#             k: np.array(v) for k, v in obj_name_to_positions.items()
#         }

#     return scene_to_obj_name_to_positions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "-d", action="store_true", default=False)
    parser.add_argument("--train_unseen", "-t", action="store_true", default=False)
    args = parser.parse_args()

    nprocesses = max(mp.cpu_count() // 2, 1)
    # nprocesses = 1

    stage_seeds = get_random_seeds()

    os.makedirs(STARTER_HOME_SERVICE_DATA_DIR, exist_ok=True)

    stage_to_task_key_to_task_orders = {stage: {} for stage in ("train", "val", "test")}
    for stage in ("train", "val", "test"):
        path = os.path.join(STARTER_HOME_SERVICE_DATA_DIR, f"{stage}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                stage_to_task_key_to_task_orders[stage] = json.load(f)
    
    send_queue = mp.Queue()
    num_task_key_to_run = 0
    for task_order in TASK_ORDERS:
        pickup_object_types = task_order["pickupObjectTypes"]
        for pickup_object_type in pickup_object_types:
            single_task_order = task_order.copy()
            single_task_order["pickupObjectTypes"] = [pickup_object_type]
            if pickup_object_type in OBJECTS_FOR_TEST:
                stage = "test"
            else:
                stage = "train"
            task_key = (
                f'Pick_{pickup_object_type}_On_{single_task_order["startReceptacleType"]}_And_Place_{single_task_order["targetReceptacleType"]}'
                if single_task_order["targetReceptacleType"] != "User" else
                f'Bring_Me_{pickup_object_type}_On_{single_task_order["startReceptacleType"]}'
            )
            # task_key = (
            #     pickup_object_type,
            #     single_task_order["startReceptacleType"],
            #     single_task_order["targetReceptacleType"],
            # )
            if task_key not in stage_to_task_key_to_task_orders[stage]:
                num_task_key_to_run += 1
                send_queue.put((single_task_order, stage, stage_seeds[stage]))

    receive_queue = mp.Queue()
    processes = []
    for i in range(nprocesses):
        p = mp.Process(
            target=rearrangement_datagen_worker,
            kwargs=dict(
                input_queue=send_queue,
                output_queue=receive_queue,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.5)

    num_received = 0
    while num_task_key_to_run > num_received:
        try:
            task_key, stage, data = receive_queue.get(timeout=1)
            num_received += 1
        except queue.Empty:
            continue

        print(f"Saving {task_key}")

        task_key_to_task_orders = stage_to_task_key_to_task_orders[stage]
        if task_key not in task_key_to_task_orders:
            task_key_to_task_orders[task_key] = []

        task_key_to_task_orders[task_key].extend(data)

        with open(os.path.join(STARTER_HOME_SERVICE_DATA_DIR, f"{stage}.json"), "w") as f:
            json.dump(task_key_to_task_orders, f)

        compress_pickle.dump(
            obj=task_key_to_task_orders,
            path=os.path.join(STARTER_HOME_SERVICE_DATA_DIR, f"{stage}.pkl.gz"),
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    for p in processes:
        try:
            p.join(timeout=1)
        except:
            pass

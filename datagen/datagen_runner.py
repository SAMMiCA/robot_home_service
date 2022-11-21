"""A script for generating robot home service datasets."""

import argparse
import json
import multiprocessing as mp
import os
import pickle
import platform
import queue
import random
import sys
import warnings
import time
import copy
from pathlib import Path
from collections import defaultdict
from typing import List, Sequence, Set, Dict, Optional, Any, cast

sys.path.insert(0, os.getcwd())

import compress_pickle
import numpy as np
import torch
import tqdm
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

from allenact.utils.misc_utils import md5_hash_str_as_int
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
# from datagen.datagen_constants import TASK_ORDERS
from datagen.datagen_utils import (
    get_random_seeds,
    find_object_by_type,
    scene_from_type_idx,
    remove_objects_until_all_have_identical_meshes,
    check_object_opens,
    filter_pickupable,
    mapping_counts,
    open_objs
)
from datagen.datagen_constants import NUM_TEST_TASKS, NUM_TRAIN_TASKS, NUM_VAL_TASKS, PICKUPABLE_RECEPTACLE_PAIRS, PICKUP_OBJECTS_FOR_TEST, GOOD_4_ROOM_HOUSES, STAGE_TO_DEST_NUM_SCENES, STAGE_TO_VALID_TASK_TO_SCENES, STAGE_TO_VALID_TASKS
from env.constants import IOU_THRESHOLD, OBJECT_TYPES_WITH_PROPERTIES, SCENE_TYPE_TO_SCENES, STARTER_HOME_SERVICE_DATA_DIR, THOR_COMMIT_ID
from env.environment import (
    HomeServiceEnvironment,
    HomeServiceTaskSpec,
)
from env.utils import extract_obj_data
from utils.multiprocessing_utils import Manager, Worker, get_logger


NUM_TRAIN_UNSEEN_EPISODES = 1_000  # 1 episode per scene
NUM_TRAIN_SCENES = 10_000  # N episodes per scene
NUM_VALID_SCENES = 1_000  # 10 episodes per scene
NUM_TEST_SCENES = 1_000  # 1 episode per scene

MAX_TRIES = 40
EXTENDED_TRIES = 10

mp = mp.get_context("spawn")

# Includes types used in both open and pickup actions:
VALID_TARGET_TYPES = {
    "AlarmClock",
    "AluminumFoil",
    "Apple",
    "BaseballBat",
    "BasketBall",
    "Blinds",
    "Book",
    "Boots",
    "Bottle",
    "Bowl",
    "Box",
    "Bread",
    "ButterKnife",
    "CD",
    "Cabinet",
    "Candle",
    "CellPhone",
    "Cloth",
    "CreditCard",
    "Cup",
    "DishSponge",
    "Drawer",
    "Dumbbell",
    "Egg",
    "Footstool",
    "Fork",
    "Fridge",
    "HandTowel",
    "Kettle",
    "KeyChain",
    "Knife",
    "Ladle",
    "Laptop",
    "LaundryHamper",
    "Lettuce",
    "Microwave",
    "Mug",
    "Newspaper",
    "Pan",
    "PaperTowelRoll",
    "Pen",
    "Pencil",
    "PepperShaker",
    "Pillow",
    "Plate",
    "Plunger",
    "Pot",
    "Potato",
    "RemoteControl",
    "Safe",
    "SaltShaker",
    "ScrubBrush",
    "ShowerCurtain",
    "ShowerDoor",
    "SoapBar",
    "SoapBottle",
    "Spatula",
    "Spoon",
    "SprayBottle",
    "Statue",
    "TableTopDecor",
    "TeddyBear",
    "TennisRacket",
    "TissueBox",
    "Toilet",
    "ToiletPaper",
    "Tomato",
    "Towel",
    "Vase",
    "Watch",
    "WateringCan",
    "WineBottle",
}


def get_scene_limits_for_task(
    env: HomeServiceEnvironment,
    # task: Dict[str, Any],
    task: str,
    scene: str,
    object_types_to_not_move: Set[str] = set(),
):
    controller = env.controller
    pick, recep = task.split("_")[-2], task.split("_")[-1]
    
    # Invalid Scene
    if not env.procthor_reset(scene_name=scene, force_reset=True):
        print(f"Cannot reset scene {scene}")
        return None

    if not remove_objects_until_all_have_identical_meshes(controller):
        print(f"Failed to remove_objects_until_all_have_identical_meshes in {scene}")
        return None

    all_objects = controller.last_event.metadata["objects"]
    if any(o["isBroken"] for o in all_objects):
        print(f"Broken objects in {scene}")
        return None

    room_reachable, reachability_meta = env.all_rooms_reachable()
    if not room_reachable:
        print(f"Unreachable rooms in {scene}: {reachability_meta}")
        return None
    
    openable_objects = env.obj_id_with_cond_to_room(
        lambda o: o["openable"]
        and not o["pickupable"]
        and o["objectType"] in VALID_TARGET_TYPES
    )

    meta_rps = controller.step("GetReachablePositions").metadata
    if meta_rps["lastActionSuccess"]:
        rps = meta_rps["actionReturn"][:]
    else:
        print(
            f"In {scene}, couldn't get reachable positions despite all rooms being reachable (?)"
        )
        return None
    
    all_objects = env.ids_to_objs()

    room_to_openable_ids = defaultdict(list)
    for oid, room in openable_objects.items():
        interactable_poses = controller.step(
            "GetInteractablePoses", objectId=oid, positions=rps,
        ).metadata["actionReturn"]
        if interactable_poses is None or len(interactable_poses) == 0:
            continue

        could_open_close, could_open, could_close = check_object_opens(
            all_objects[oid], controller, return_open_closed=True
        )
        if not could_close:
            if could_open:
                print(f"Couldn't close {oid} fully in {scene}")
                return None
            continue
        if could_open_close:
            room_to_openable_ids[room].append(oid)

    pickupable_objects = filter_pickupable(
        objects=[
            all_objects[obj]
            for obj in all_objects
            if all_objects[obj]["objectType"] in VALID_TARGET_TYPES
        ],
        object_types_to_not_move=object_types_to_not_move
    )

    if len(pickupable_objects) == 0:
        print(f"No objects to pickup in {scene}")
        return None

    # Check Pickup target existence
    pickup_object = next(
        (
            obj 
            for obj in pickupable_objects
            if obj["objectType"] == pick
        ),
        None
    )
    if pickup_object is None:
        print(f'No {pick} in {scene}.')
        return None
        
    receps_per_room = {
        room: env.static_receptacles_in_room(room) for room in env.room_to_poly
    }

    for room, rids in receps_per_room.items():
        reachable_ids = []
        for rid in rids:
            interactable_poses = controller.step(
                "GetInteractablePoses", objectId=rid, positions=rps,
            ).metadata["actionReturn"]
            if interactable_poses is None or len(interactable_poses) == 0:
                continue
            else:
                reachable_ids.append(rid)
        receps_per_room[room] = reachable_ids

    # Check target receptacle existence
    receps = []
    for room, rids in receps_per_room.items():
        receps.extend(
            [
                rid
                for rid in rids
                if all_objects[rid]["objectType"] == recep
            ]
        )
    
    # Only 1 recep should exist in the house...
    if (
        recep != "User"
        and len(receps) != 1
    ):
        if len(receps) == 0:
            print(f'No {recep} in {scene}')
        else:
            print(f'More than 1 {recep}s in {scene}')
        return None

    # num_receps_per_room = mapping_counts(receps_per_room)
    # if any(v < 2 for v in num_receps_per_room.values()):
    #     print(
    #         f"Less than 2 receptacles in some room(s) in {scene}: {num_receps_per_room}"
    #     )
    #     return None

    return dict(
        pickupables=pickupable_objects,             # In entire house
        room_openables={**room_to_openable_ids},    # in each room
        room_receptacles={**receps_per_room},            # in each room
    )

def generate_one_home_service_given_initial_conditions(
    env: HomeServiceEnvironment,
    scene: str,
    task: str,
    init_kwargs: dict,
    agent_pos: Dict[str, float],
    agent_rot: Dict[str, float],
    starting_room: str,
    target_room: str,
    # single_room: str,
    # object_types_to_not_move: Set[str],
    # allow_putting_objects_away: bool = False,
    # possible_openable_ids: Optional[List[str]] = None,
):
    controller = env.controller
    pick, recep = task.split("_")[-2], task.split("_")[-1]

    env.procthor_reset(scene_name=scene, force_reset=True)
    controller.step(
        "TeleportFull",
        horizon=0,
        standing=True,
        rotation=agent_rot,
        **agent_pos,
    )
    if not controller.last_event.metadata["lastActionSuccess"]:
        print(controller.last_event.metadata["errorMessage"])
        return None, None, None

    if not remove_objects_until_all_have_identical_meshes(controller):
        print("Error initially removing objects")
        return None, None, None

    # Works around a labeling issue in THOR for floor lamps (pickupable)
    excluded_object_ids_for_floor_lamp = [
        o["objectId"]
        for o in controller.last_event.metadata["objects"]
        if o["objectType"] == "FloorLamp"
    ]

    controller.step(
        "InitialRandomSpawn",
        **{
            **init_kwargs, 
            "excludedReceptacles": init_kwargs["excludedReceptacles"].append(recep)
        },
        excludedObjectIds=excluded_object_ids_for_floor_lamp,
    )
    if not controller.last_event.metadata["lastActionSuccess"]:
        print(controller.last_event.metadata["errorMessage"])
        return None, None, None

    for _ in range(12):
        controller.step("Pass")

    # get initial and post random spawn object data
    objects_after_first_irs = copy.deepcopy(env.objects())

    if any(o["isBroken"] for o in objects_after_first_irs):
        print("Broken objects after first irs")
        return None, None, None

    # What if opening moved a pickupable? Get the updated list!
    objects_after_first_irs = copy.deepcopy(env.objects())

    if any(o["isBroken"] for o in objects_after_first_irs):
        print("Broken objects after opening")
        return None, None, None

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
            if obj["objectType"] == pick
        ),
        None
    )
    if pickup_target is None:
        print(f"there is no pickup target {pick} in the scene {scene}")
        return None, None, None

    recep_start = pickup_target["parentReceptacles"]
    if recep_start is not None:
        if recep in [
            robj["objectType"]
            for rid, robj in env.ids_to_objs(source=objects_after_first_irs).items()
            if rid in recep_start
        ]:
            print(f"pickup target {pick} is already placed on the recep target {recep}")
            return None, None, None

    # Ensure that the pickup target is interactable
    rps = controller.step("GetReachablePositions").metadata["actionReturn"][:]
    pickup_interactable_poses = controller.step(
        "GetInteractablePoses",
        objectId=pickup_target["objectId"],
        positions=rps,
    ).metadata["actionReturn"]
    if pickup_interactable_poses is None or len(pickup_interactable_poses) == 0:
        print(f"pickup target {pick} is spawned at not interactable position...")
        return None, None, None

    controller.step(
        "TeleportFull", horizon=0, standing=True, rotation=agent_rot, **agent_pos
    )
    if not controller.last_event.metadata["lastActionSuccess"]:
        print(controller.last_event.metadata["errorMessage"])
        return None, None, None

    # Ensure the recep target 
    objs_to_open = []
    if recep != "User":
        recep_targets = [
            env.ids_to_objs(source=objects_after_first_irs)[obj]
            for obj in env.static_receptacles_in_room(room_id=target_room)
            if env.ids_to_objs(source=objects_after_first_irs)[obj]["objectType"] == recep
        ]
        if len(recep_targets) == 0:
            print(f"there is no place recep target {recep} in the scene {scene}")
            return None, None, None

        st = random.getstate()
        random.seed(7654321)
        random.shuffle(recep_targets)
        random.setstate(st)

        for recep_target in recep_targets:
            # Open recep_target
            if recep_target["openable"]:
                start_openness = recep_target["openness"]
                if not check_object_opens(recep_target, controller):
                    print(f"Failed to open & close recep {recep} in the scene {scene}")
                    return None, None, None

                controller.step(
                    "OpenObject",
                    objectId=recep_target["objectId"],
                    openness=1.0,
                    forceAction=True,
                )
                
                # What if opening moved a pickupable? Get the updated list!
                objects_after_first_irs = copy.deepcopy(env.objects())

                if any(o["isBroken"] for o in objects_after_first_irs):
                    print("Broken objects after opening")
                    return None, None, None

            # Place the pickup target above the recep target to examine the possibility
            controller.step(
                "GetSpawnCoordinatesAboveReceptacle",
                objectId=recep_target["objectId"],
                anywhere=True,
            )
            if not controller.last_event.metadata["lastActionSuccess"]:
                print(controller.last_event.metadata["errorMessage"])
                return None, None, None
        
            recep_pts = controller.last_event.metadata["actionReturn"][:]
            random.shuffle(recep_pts)

            while len(recep_pts) > 0:
                controller.step(
                    "PlaceObjectAtPoint",
                    objectId=pickup_target["objectId"],
                    position=recep_pts.pop(),
                )
                if not controller.last_event.metadata["lastActionSuccess"]:
                    continue
                else:
                    objects_after_check_receps = copy.deepcopy(env.objects())
                    pickupable_objects_after_check_receps = [
                        obj
                        for obj in objects_after_check_receps
                        if obj["pickupable"]
                    ]
                    for o in pickupable_objects_after_check_receps:
                        if o["isBroken"]:
                            print(
                                f"In scene {scene} object {o['objectId']} broke during setup."
                            )
                            return None, None, None
                        if o["objectType"] == pick:
                            pickup_target_after_check_recep = o
                    
                    o_pos = pickup_target_after_check_recep["position"]
                    check_positions = [
                        {
                            "x": o_pos["x"] + 0.001 * x_off,
                            "y": o_pos["y"] + 0.001 * y_off,
                            "z": o_pos["z"] + 0.001 * z_off,
                        }
                        for x_off in [0, -1, 1]
                        for y_off in [0, 1, 2]
                        for z_off in [0, -1, 1]
                    ]
                    controller.step(
                        "TeleportObject",
                        objectId=pickup_target_after_check_recep["objectId"],
                        positions=check_positions,
                        rotation=pickup_target_after_check_recep["rotation"],
                        makeUnbreakable=True,
                    )
                    if not controller.last_event.metadata["lastActionSuccess"]:
                        continue

                    break
        
            if not controller.last_event.metadata["lastActionSuccess"]:
                print(f"No possible points to place the target {pick} on recep {recep_target['objectId']}.")
                # reset openness of the current recep target
                if recep_target["openable"]:
                    controller.step(
                        "OpenObject",
                        objectId=recep_target["objectId"],
                        openness=start_openness,
                        forceAction=True,
                    )
                continue
            else:
                if recep_target["openable"]:
                    objs_to_open.append(recep_target)
                break
        
        if not controller.last_event.metadata["lastActionSuccess"]:
            print(f"No possible candidate for recep {recep} to place the target {pick} in the scene {scene}.")
            return None, None, None

    else:
        # Agent holding pickup target at the starting position
        controller.step(
            "PickupObject",
            objectId=pickup_target["objectId"],
            forceAction=True,
        )
        if not controller.last_event.metadata["lastActionSuccess"]:
            print(f"The agent failed to hold the pickup target {pick}")
            return None, None, None

        objects_after_check_receps = copy.deepcopy(env.objects())
        pickupable_objects_after_check_receps = [
            obj
            for obj in objects_after_check_receps
            if obj["pickupable"]
        ]

    # Ensure all rooms are reachable after the irs from the starting positions
    controller.step(
        "TeleportFull", horizon=0, standing=True, rotation=agent_rot, **agent_pos,
    )
    if not controller.last_event.metadata["lastActionSuccess"]:
        print(controller.last_event.metadata["errorMessage"])
        return None, None, None

    room_reachable, reachability_meta = env.all_rooms_reachable()
    if not room_reachable:
        print(f"Unreachable rooms in {scene} after first irs: {reachability_meta}")
        return None, None, None
    
    pickupable_objects_after_first_irs.sort(key=lambda x: x["objectId"])
    pickupable_objects_after_check_receps.sort(key=lambda x: x["objectId"])
    if any(
        o0["objectId"] != o1["objectId"]
        for o0, o1 in zip(pickupable_objects_after_first_irs, pickupable_objects_after_check_receps)
    ):
        print("Pickupable object ids don't match after shuffle!")
        return None, None, None

    for o0, o1 in zip(pickupable_objects_after_first_irs, pickupable_objects_after_check_receps):
        if o0["objectId"] == pickup_target["objectId"]:
            continue
        o0["position"] = o1["position"]
        o0["rotation"] = o1["rotation"]

    # (starting, target, open)
    return (
        [
            {
                "name": pickupable_objects_after_first_irs[i]["name"],
                "objectName": pickupable_objects_after_first_irs[i]["name"],
                "position": pickupable_objects_after_first_irs[i]["position"],
                "rotation": pickupable_objects_after_first_irs[i]["rotation"],
            }
            for i in range(len(pickupable_objects_after_first_irs))
        ],
        [
            {
                "name": pickupable_objects_after_check_receps[i]["name"],
                "objectName": pickupable_objects_after_check_receps[i]["name"],
                "position": pickupable_objects_after_check_receps[i]["position"],
                "rotation": pickupable_objects_after_check_receps[i]["rotation"],
            }
            for i in range(len(pickupable_objects_after_check_receps))
        ],
        [
            {
                **obj,
            }
            for obj in objs_to_open
        ],
    )

def generate_home_service_episode_for_task(
    stage_seed: int,
    scene: str,
    env: HomeServiceEnvironment,
    house_count: int,
    scene_count: int,
    task: str,
    stage: str = "train",
    house_i: int = 0,
    scene_i: int = 0,
    force_visible: bool = True,
    place_stationary: bool = False,
    rotation_increment: int = 90,
    allow_moveable_in_goal_randomization: bool = False,
    limits: Optional[Dict[str, Any]] = None,
    object_types_to_not_move: Set[str] = set(),
    # obj_name_to_avoid_positions=obj_name_to_avoid_positions,
) -> Optional[Any]:
    if 360 % rotation_increment != 0:
        raise ValueError("Rotation increment must be a factor of 360")

    # if obj_name_to_avoid_positions is None:
    #     obj_name_to_avoid_positions = defaultdict(
    #         lambda: np.array([[-1000, -1000, -1000]])
    #     )

    controller = env.controller
    pick, recep = task.split("_")[-2], task.split("_")[-1]

    print(f"Task {stage} {task} for {scene}")

    seed = md5_hash_str_as_int(f"{stage_seed}|{task}|{scene}")
    random.seed(seed)

    if limits is None:
        print(f"Re-computing limits for Task {task} {stage} {scene} {house_i} {scene_i}???????")
        if env.num_rooms(scene) != 4:
            print(f"{scene} has {env.num_rooms(scene)} rooms. Skipping.")
            return None
        
        # House loaded in get_scene_limits
        limits = get_scene_limits_for_task(
            env,
            task,
            scene,
            object_types_to_not_move,
        )
        if limits is None:
            print(f"Cannot use scene {scene}.")
            return None
    
    else:
        if not env.procthor_reset(scene_name=scene, force_reset=True):
            print(f"Cannot reset scene {scene}")
            return None

        if not remove_objects_until_all_have_identical_meshes(controller):
            print(
                f"Failed to remove_objects_until_all_have_identical_meshes in {scene}"
            )
            return None

    scene_has_openable = 0 != len(
        [
            o
            for o in controller.last_event.metadata["objects"]
            if o["openable"] and not o["pickupable"]
        ]
    )
    if not scene_has_openable:
        warnings.warn(f"HOUSE {scene} HAS NO OPENABLE OBJECTS")

    evt = controller.step("GetReachablePositions")
    rps: List[Dict[str, float]] = evt.metadata["actionReturn"][:]
    rps.sort(key=lambda d: (round(d["x"], 2), round(d["z"], 2)))
    rotations = np.arange(0, 360, rotation_increment)

    room_to_rps = copy.deepcopy(env.room_to_reachable_positions())
    for room, room_rps in room_to_rps.items():
        room_rps.sort(key=lambda d: (round(d["x"], 2), round(d["z"], 2)))

    assert house_i < house_count
    assert scene_i < scene_count

    try_count = 0
    # TODO:
    # Assign start room type & start receptacle type
    # If target_receptacle type is openable, it should be opened for '''easy''' tasks
    # define possible_room

    position_count_offset = 0
    # room that has receptacle in it
    all_objects = env.ids_to_objs()
    possible_starting_rooms = sorted(list(env.room_to_poly.keys()))
    possible_target_rooms = []
    if recep != "User":
        possible_target_rooms = [
            room_id
            for room_id, rids in limits["room_receptacles"].items()
            for rid in rids
            if all_objects[rid]["objectType"] == recep
        ]

    st = random.getstate()
    random.seed(1234567)
    random.shuffle(possible_starting_rooms)
    random.shuffle(possible_target_rooms)
    random.setstate(st)

    while True:
        try_count += 1
        if try_count > MAX_TRIES + EXTENDED_TRIES:
            print(f"Something wrong with house {scene} scene_i {scene_i}. Skipping")
            return None
        if try_count == MAX_TRIES + 1:
            print(f"Something wrong with house {scene} scene_i {scene_i}. Trying another room.")
            if len(set(possible_starting_rooms)) > 1 or len(set(possible_target_rooms)) > 1:
                if len(set(possible_starting_rooms)) > 1:
                    possible_starting_rooms = [r for r in possible_starting_rooms if r != starting_room]
                if len(set(possible_target_rooms)) > 1:
                    possible_target_rooms = [r for r in possible_target_rooms if r != target_room]
            else:
                return None

        episode_seed_string = f"{task}|{scene}|ind_{scene_i}|tries_{try_count}|counts_{position_count_offset}|seed_{stage_seed}"
        seed = md5_hash_str_as_int(episode_seed_string)
        random.seed(seed)

        starting_room = cast(str, possible_starting_rooms[scene_i % len(possible_starting_rooms)])
        if recep != "User":
            target_room = cast(str, possible_target_rooms[scene_i % len(possible_target_rooms)])
        else:
            target_room = starting_room
        
        # avoid agent being unable to teleport to position
        # due to object being placed there
        pos = random.choice(room_to_rps[starting_room])
        rot = {"x": 0, "y": int(random.choice(rotations)), "z": 0}

        # used to make sure the positions of the objects
        # are not always the same across the same scene.
        init_kwargs = {
            "randomSeed": random.randint(0, int(1e7) - 1),
            "forceVisible": force_visible,
            "placeStationary": place_stationary,
            "excludedReceptacles": ["ToiletPaperHanger"],
            "allowMoveable": allow_moveable_in_goal_randomization,
        }

        (
            starting_poses,
            target_poses,
            objs_to_open,
        ) = generate_one_home_service_given_initial_conditions(
            env=env,
            scene=scene,
            task=task,
            init_kwargs=init_kwargs,
            agent_pos=pos,
            agent_rot=rot,
            starting_room=starting_room,
            target_room=target_room,
            # single_room=single_room,
            # object_types_to_not_move=object_types_to_not_move,
            # allow_putting_objects_away=MAX_TRIES >= try_count >= MAX_TRIES // 2,
            # possible_openable_ids=limits["room_openables"][single_room]
            # if single_room in limits["room_openables"]
            # else [],
        )
        
        if starting_poses is None or target_poses is None:
            print(f"{episode_seed_string}: Failed during generation...")
            continue
        
        task_spec_dict = {
            "pickup_object": pick,
            "target_receptacle": recep,
            "agent_position": pos,
            "agent_rotation": int(rot["y"]),
            "starting_poses": starting_poses,
            "target_poses": target_poses,
            "objs_to_open": objs_to_open,
        }

        try:
            for _ in range(1):
                env.reset(
                    task_spec=HomeServiceTaskSpec(
                        scene=scene,
                        **task_spec_dict,
                    ),
                    raise_on_inconsistency=True,
                )
                assert env.all_rooms_reachable()[0]
        except:
            get_logger().info(
                f"{episode_seed_string}: Inconsistency or room unreachability when reloading task spec."
            )
            continue
    
        ips, gps, cps = env.poses
        pose_diffs = cast(
            List[Dict[str, Any]], env.compare_poses(goal_pose=gps, cur_pose=cps)
        )
        reachable_positions = controller.step("GetReachablePositions").metadata[
            "actionReturn"
        ]
        # cps == ips
        failed = False
        for gp, cp, pd in zip(gps, cps, pose_diffs):
            if pd["iou"] is not None and pd["iou"] < IOU_THRESHOLD:
                if gp["type"] != pick and gp["type"] != recep:
                    failed = True
                    print(
                        f"{episode_seed_string}: Moved object ({gp['type']}) not pick [{pick}] nor recep [{recep}]."
                    )
                    break
                
            if gp["broken"] or cp["broken"]:
                failed = True
                print(f"{episode_seed_string}: Broken object")
                break

            pose_diff_energy = env.pose_difference_energy(goal_pose=gp, cur_pose=cp)

            if pose_diff_energy != 0:
                obj_name = gp["objectId"]

                # Ensure that objects to rearrange are visible from somewhere
                interactable_poses = env.controller.step(
                    "GetInteractablePoses",
                    objectId=cp["objectId"],
                    positions=reachable_positions,
                ).metadata["actionReturn"]
                if interactable_poses is None or len(interactable_poses) == 0:
                    print(
                        f"{episode_seed_string}: {obj_name} is not visible despite needing to be rearranged."
                    )

                    failed = True
                    break
        
        if failed:
            continue

        task_spec_dict["pose_diff_energy"] = float(
            env.pose_difference_energy(goal_pose=gps, cur_pose=cps).sum()
        )

        if task_spec_dict["pose_diff_energy"] == 0.0:
            print(f"Not moved...")
            continue
    
        print(f"{episode_seed_string} SUCCESS")
        return task_spec_dict


def find_scene_to_limits_for_task(
    stage_seed: int,
    scene: str,
    env: HomeServiceEnvironment,
    stage: str,
    task: str,
    house_count: int,
    force_visible: bool = True,
    place_stationary: bool = False,
    rotation_increment: int = 90,
    allow_moveable_in_goal_randomization: bool = False,
    # object_types_to_not_move=,
    # obj_name_to_avoid_positions=obj_name_to_avoid_positions,
) -> Optional[Any]:
    if 360 % rotation_increment != 0:
        raise ValueError("Rotation increment must be a factor of 360")

    scene_to_limits = {}
    if stage == "debug":
        stage = task.split("_")[0]

    shuffled_scenes = copy.deepcopy(STAGE_TO_VALID_TASK_TO_SCENES[stage][task])

    st = random.getstate()
    random.seed(stage_seed)
    random.shuffle(shuffled_scenes)
    random.setstate(st)

    for scene in shuffled_scenes:
        print(f"Task {stage} {task} for {scene} limits")

        if env.num_rooms(scene) != 4:
            print(f"{scene} has {env.num_rooms(scene)} rooms. Skipping.")
            continue
    
        # House loaded in get_scene_limits_for_task
        limits = get_scene_limits_for_task(
            env,
            task,
            scene,
            # object_types_to_not_move=object_types_to_not_move,
        )
        if limits is None:
            print(f"Cannot use scene {scene} for task {task}.")
            continue
        
        else:
            scene_to_limits[scene] = limits
            if len(scene_to_limits) == house_count:
                break
    
    if len(scene_to_limits) < house_count:
        print(f"Task {stage} {task} not enough num_scenes. Skipping Task.")
        return None

    return scene_to_limits


class HomeServiceDatagenWorker(Worker):

    def create_env(self, **env_args: Any) -> Optional[Any]:
        env = HomeServiceEnvironment(
            force_cache_reset=True,
            controller_kwargs={
                "scene": "Procedural",
                "x_display": f"0.{self.gpu}" if self.gpu is not None else None,
                "platform": CloudRendering,
            },
        )

        return env

    def work(self, task_type: Optional[str], task_info: Dict[str, Any]) -> Optional[Any]:
        if task_type == "home_service":
            (
                scene,
                seed,
                stage,
                task,
                house_i,
                scene_i,
                limits,
                house_count,
                scene_count,
                # obj_name_to_avoid_positions,
            ) = (
                task_info["scene"],
                task_info["seed"],
                task_info["stage"],
                task_info["task"],
                task_info["house_i"],
                task_info["scene_i"],
                task_info["limits"],
                task_info["house_count"],
                task_info["scene_count"],
                # task_info["obj_name_to_avoid_positions"],
            )

            mode, idx = tuple(scene.split("_"))
            self.env._houses.mode(mode)

            data = generate_home_service_episode_for_task(
                stage_seed=seed,
                scene=scene,
                env=self.env,
                task=task,
                stage=stage,
                house_count=house_count,
                scene_count=scene_count,
                house_i=house_i,
                scene_i=scene_i,
                limits=limits
                # object_types_to_not_move=,
                # obj_name_to_avoid_positions=obj_name_to_avoid_positions,
            )

            return data

        elif task_type == "find_limits":
            (
                scene,
                seed,
                stage,
                task,
                house_count,
                # scene_i,
                # obj_name_to_avoid_positions,
            ) = (
                task_info["scene"],
                task_info["seed"],
                task_info["stage"],
                task_info["task"],
                task_info["house_count"],
                # task_info["scene_i"],
                # task_info["obj_name_to_avoid_positions"],
            )

            mode = task.split("_")[0]
            self.env._houses.mode(mode)

            data = find_scene_to_limits_for_task(
                stage_seed=seed,
                scene=scene,
                env=self.env,
                stage=stage,
                task=task,
                house_count=house_count,
                # house_i=house_i,
                # scene_i=scene_i,
                # object_types_to_not_move=,
                # obj_name_to_avoid_positions=obj_name_to_avoid_positions,
            )

            return data


def args_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "-d", action="store_true", default=False)
    parser.add_argument("--train_unseen", "-t", action="store_true", default=False)
    parser.add_argument("--mode", "-m", default="train")
    return parser.parse_args()


class HomeServiceDatagenManager(Manager):

    def work(
        self,
        task_type: Optional[str],
        task_info: Dict[str, Any],
        success: bool,
        result: Any
    ) -> None:
        if task_type is None:
            args = args_parsing()

            stage_seeds = get_random_seeds()

            # max_scene_count: number of houses for each task type
            self.house_count = STAGE_TO_DEST_NUM_SCENES[args.mode]
            if args.mode == "train":
                self.scene_count = 5
            elif args.mode in ["val", "test"]:
                self.scene_count = 2

            # stage_to_tasks = {
            #     stage: [
            #         f"{stage}_pick_and_place_{pick}_{recep}"
            #         for pick, recep in PICKUPABLE_RECEPTACLE_PAIRS
            #         if pick not in (PICKUP_OBJECTS_FOR_TEST if 'train' in stage else [])
            #     ]
            #     for stage in [args.mode]
            # }
            stage_to_tasks = {
                stage: STAGE_TO_VALID_TASKS[stage]
                for stage in [args.mode]
            }

            # scene_to_obj_name_to_avoid_positions = None
            if args.debug:
                partition = "train" if args.mode == "train" else "valid"
                idxs = [0, 1, 2]
                self.house_count = 3
                self.scene_count = 2
                stage_to_tasks = {
                    "debug": [
                        stage_to_tasks[partition][idx]
                        for idx in idxs
                    ]
                }

            os.makedirs(STARTER_HOME_SERVICE_DATA_DIR, exist_ok=True)

            self.last_save_time = {stage: time.time() for stage in stage_to_tasks}

            self.stage_to_task_to_task_specs = {
                stage: {} for stage in stage_to_tasks
            }
            for stage in stage_to_tasks:
                path = os.path.join(STARTER_HOME_SERVICE_DATA_DIR, f"{stage}.json")
                if os.path.exists(path):
                    with open(path, "r") as f:
                        self.stage_to_task_to_task_specs[stage] = json.load(f)
            
            for stage in stage_to_tasks:
                for task in stage_to_tasks[stage]:
                    if task not in self.stage_to_task_to_task_specs[stage]:
                        self.stage_to_task_to_task_specs[stage][task] = [
                            [-1] * self.scene_count
                        ] * self.house_count
                        self.enqueue(
                            task_type="find_limits",
                            task_info=dict(
                                scene="",
                                stage=stage,
                                task=task,
                                seed=stage_seeds[stage],
                                house_count=self.house_count,
                                # scene_i=-1,
                                # obj_name_to_avoid_positions=obj_name_to_avoid_positions,
                            )
                        )

        elif task_type == "find_limits":
            if result is not None:
                for it, (scene, limits) in enumerate(result.items()):
                    task_info["scene"] = scene
                    task_info["limits"] = limits
                    for scene_i in range(self.scene_count):
                        if (
                            self.stage_to_task_to_task_specs[task_info["stage"]][task_info["task"]][it][scene_i] == -1
                        ):
                            task_info["house_i"] = it
                            task_info["scene_i"] = scene_i
                            task_info["house_count"] = self.house_count
                            task_info["scene_count"] = self.scene_count
                            self.enqueue(
                                task_type="home_service",
                                task_info=copy.deepcopy(task_info)
                            )
            else:
                # result is None, delete task from data
                del self.stage_to_task_to_task_specs[task_info["stage"]][task_info["task"]]
        
        elif task_type == "home_service":
            scene, stage, task, seed, house_i, scene_i = (
                task_info["scene"],
                task_info["stage"],
                task_info["task"],
                task_info["seed"],
                task_info["house_i"],
                task_info["scene_i"],
            )

            task_to_task_specs = self.stage_to_task_to_task_specs[stage]
            task_to_task_specs[task][house_i][scene_i] = result

            num_missing = len(
                [
                    ep
                    for houses in task_to_task_specs[task]
                    for ep in houses
                    if ep == -1
                ]
            )

            if num_missing == 0:
                get_logger().info(
                    f"{self.info_header}: Completed {stage} {task}"
                )

            for stage in self.last_save_time:
                if self.all_work_done or (
                    time.time() - self.last_save_time[stage] > 30 * 60
                ):
                    get_logger().info(self.info_header + f": Saving {stage}")

                    with open(
                        os.path.join(STARTER_HOME_SERVICE_DATA_DIR, f"{stage}.json"), "w"
                    ) as f:
                        json.dump(self.stage_to_task_to_task_specs[stage], f)

                    compress_pickle.dump(
                        obj=self.stage_to_task_to_task_specs[stage],
                        path=os.path.join(STARTER_HOME_SERVICE_DATA_DIR, f"{stage}.pkl.gz"),
                        # pickler_kwargs={
                        #     "protocol": 4,
                        # },  # Backwards compatible with python 3.6
                    )

                    self.last_save_time[stage] = time.time()

        else:
            raise ValueError(f"Unknown task type {task_type}")

if __name__ == "__main__":
    args = args_parsing()

    print(f"Using args {args}")

    assert args.mode in ["train", "val", "test"]
    if args.train_unseen:
        assert args.mode == "train"

    HomeServiceDatagenManager(
        worker_class=HomeServiceDatagenWorker,
        env_args={},
        workers=max((2 * mp.cpu_count()) // 4, 1)
        if platform.system() == "Linux" and not args.debug
        else 1,
        ngpus=torch.cuda.device_count(),
        die_on_exception=False,
        verbose=True,
        debugging=args.debug,
        sleep_between_workers=1.0,
    )
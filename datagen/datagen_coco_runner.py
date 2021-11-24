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
from coco_data_collection import HOME_SERVICE_OBJECTS
from datagen.datagen_constants import OBJECTS_FOR_TEST, TASK_ORDERS
from datagen.datagen_utils import (
    get_random_seeds,
    find_object_by_type,
    scene_from_type_idx
)
from env.constants import OBJECT_TYPES_WITH_PROPERTIES, SCENE_TYPE_TO_SCENES, STARTER_HOME_SERVICE_DATA_DIR, THOR_COMMIT_ID
from env.environment import (
    HomeServiceTHOREnvironment,
    HomeServiceTaskSpec,
)

mp = mp.get_context("spawn")

def generate_task_specs_for_task_orders(
    stage_seed: int,
    stage_scenes: List[str],
    env: HomeServiceTHOREnvironment,
    object_types_to_remove: Set[str],
    scene_inds: Sequence[int],
    force_visible: bool = True,
    place_stationary: bool = False,
    rotation_increment: int = 30,
) -> dict:
    if 360 % rotation_increment != 0:
        raise ValueError("Rotation increment must be a factor of 360")

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
            task_key = (
                f'Pick_{pickup_object_type}_On_{start_receptacle_type}_And_Place_{target_receptacle_type}'
                if target_receptacle_type != "User" else
                f'Bring_Me_{pickup_object_type}_On_{start_receptacle_type}'
            )
            out[task_key] = []
            for i in scene_inds:                   
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
                if target_receptacle_type != "User":
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
    return out

def rearrangement_datagen_worker(
    input_queue: mp.Queue,
    output_queue: mp.Queue,
):
    env = HomeServiceTHOREnvironment(
        force_cache_reset=True, controller_kwargs={"commit_id": THOR_COMMIT_ID}
    )

    while True:
        try:
            single_task_order, stage, scene_inds, seed = input_queue.get(timeout=2)
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
            scene_inds=scene_inds,
            env=env,
        )
        output_queue.put((key, stage, data[key]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "-d", action="store_true", default=False)
    parser.add_argument("--train_unseen", "-t", action="store_true", default=False)
    args = parser.parse_args()

    nprocesses = max(mp.cpu_count() // 2, 1)
    # nprocesses = 1

    stage_seeds = get_random_seeds()

    os.makedirs(STARTER_HOME_SERVICE_DATA_DIR, exist_ok=True)

    stage_to_task_key_to_task_orders = {stage: {} for stage in ("train_seen", "train_unseen", "test_seen", "test_unseen")}
    for stage in ("train_seen", "train_unseen", "test_seen", "test_unseen"):
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
            task_key = (
                f'Pick_{pickup_object_type}_On_{single_task_order["startReceptacleType"]}_And_Place_{single_task_order["targetReceptacleType"]}'
                if single_task_order["targetReceptacleType"] != "User" else
                f'Bring_Me_{pickup_object_type}_On_{single_task_order["startReceptacleType"]}'
            )
            for room_seen in ("seen", "unseen"):
                if pickup_object_type in OBJECTS_FOR_TEST:
                    stage = "test"
                else:
                    stage = "train"
                stage = "_".join([stage, room_seen])
                if room_seen == "seen":
                    scene_inds = range(1, 21)
                elif room_seen == "unseen":
                    scene_inds = range(21, 31)
                else:
                    raise RuntimeError

                if task_key not in stage_to_task_key_to_task_orders[stage]:
                    num_task_key_to_run += 1
                    send_queue.put((single_task_order, stage, scene_inds, stage_seeds[stage]))

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

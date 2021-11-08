import copy
from env.environment import HomeServiceTHOREnvironment, HomeServiceTaskSpec
from env.constants import (
    DEFAULT_COMPATIBLE_RECEPTACLES,
    MAX_HAND_METERS,
    STARTER_REARRANGE_DATA_DIR, 
    STARTER_HOME_SERVICE_SIMPLE_PICK_AND_PLACE_DATA_DIR, 
    THOR_COMMIT_ID
)
from env.tasks import HomeServiceTaskSampler
from env.utils import include_object_data

import argparse
import json
import multiprocessing as mp
import os
import pickle
import queue
import random
import time
from collections import defaultdict
from typing import List, Set, Dict, Optional, Any, cast
import compress_pickle
import numpy as np
import tqdm

from datagen.datagen_utils import (
    get_scenes,
    get_random_seeds,
    filter_pickupable,
    open_objs,
    get_object_ids_to_not_move_from_object_types,
    remove_objects_until_all_have_identical_meshes,
)

mp = mp.get_context("spawn")

def generate_home_service_simple_pick_and_place_for_scenes(
    env: HomeServiceTHOREnvironment,
    scene: str,
    task_spec_dicts: List[Dict[str, Any]],
):
    controller = env.controller
    out: dict = dict()

    out[scene] = []
    # controller.reset(scene)
    for task_spec in task_spec_dicts:
        env.reset(task_spec=HomeServiceTaskSpec(**task_spec))
        scene_possible_pick_place_tasks = dict()
        with include_object_data(controller):
            objects = [obj for obj in env.last_event.metadata["objects"]]
        pickupable_object_types = {
            obj['objectType'] for obj in objects 
            if obj["pickupable"]
        }

        receptacle_object_types = {
            obj['objectType'] for obj in objects 
            if obj["receptacle"] and not obj["pickupable"]
        }

        pickup_sample = [
            obj for obj in objects 
            if (
                len(env._interactable_positions_cache.get(
                    scene_name=scene,
                    obj=obj,
                    controller=controller,
                    max_distance=MAX_HAND_METERS,
                )) > 0 and (
                    obj["objectType"] in pickupable_object_types
                )
            )
        ]

        for pick in pickup_sample:
            if pick['objectType'] in DEFAULT_COMPATIBLE_RECEPTACLES:
                place_sample = [
                    obj for obj in objects
                    if obj["objectType"] in receptacle_object_types
                    and obj["objectType"] in DEFAULT_COMPATIBLE_RECEPTACLES[pick['objectType']]
                    and len(env._interactable_positions_cache.get(
                            scene_name=scene, obj=obj, controller=controller, max_distance=MAX_HAND_METERS,
                            )) > 0
                ]
            else:
                place_sample = [
                    obj for obj in objects
                    if obj["objectType"] in receptacle_object_types
                    and len(env._interactable_positions_cache.get(
                            scene_name=scene, obj=obj, controller=controller, max_distance=MAX_HAND_METERS,
                            )) > 0
                ]
            
            if pick["parentReceptacles"] is not None:
                recep_types = [
                    o["objectType"] for o in objects 
                    for recep in pick["parentReceptacles"]
                    if o["objectId"] == recep
                ]
                place_sample = [
                    o for o in objects
                    if o["objectType"] not in recep_types
                ]

            event = controller.step("PickupObject", forceAction=True, objectId=pick["objectId"])
            assert event.metadata["lastActionSuccess"]
            for place in place_sample:
                task_type = f"Pick_{pick['objectType']}_And_Place_{place['objectType']}"
                if task_type not in scene_possible_pick_place_tasks:
                    scene_possible_pick_place_tasks[task_type] = []
                
                event = controller.step("PutObject", forceAction=True, objectId=place["objectId"])
                if event.metadata["lastActionSuccess"]:
                    pick_target = {
                        "name": pick["name"],
                        "objectName": pick["name"],
                        "objectId": pick["objectId"],
                        "objectType": pick["objectType"],
                        "parentReceptacles": pick["parentReceptacles"],
                        "position": pick["position"],
                        "rotation": pick["rotation"],
                    }

                    place_target = {
                        "name": place["name"],
                        "objectName": place["name"],
                        "objectId": place["objectId"],
                        "objectType": place["objectType"],
                        "receptacleObjectIds": place["receptacleObjectIds"],
                        "position": place["position"],
                        "rotation": place["rotation"],
                    }
                    scene_possible_pick_place_tasks[task_type].append(
                        (pick_target, place_target)
                    )

                    event = controller.step("PickupObject", forceAction=True, objectId=pick["objectId"])
                    assert event.metadata["lastActionSuccess"]
            
            event = controller.step("PausePhysicsAutoSim")
            assert event.metadata["lastActionSuccess"]

            event = controller.step(
                "TeleportObject", 
                objectId=pick["objectId"], 
                rotation=pick["rotation"],
                **pick["position"],
                forceAction=True,
                allowTeleportOutOfHand=True,
                forceKinematic=True,
            )
            assert event.metadata["lastActionSuccess"]
        
        out[scene].append(scene_possible_pick_place_tasks)
    return out

# def generate_home_service_simple_pick_and_place_type_for_scenes(
#     env: HomeServiceTHOREnvironment,
#     stage_scenes: List[str],
# ):
#     controller = env.controller
#     out: dict = dict()

#     for scene in stage_scenes:
#         out[scene] = []
#         controller.reset(scene)
#         scene_possible_pick_place_tasks = []
#         with include_object_data(controller):
#             objects = [obj for obj in env.last_event.metadata["objects"]]
        
#         pickupable_object_types = {
#             obj['objectType'] for obj in objects 
#             if obj["pickupable"]
#         }

#         receptacle_object_types = {
#             obj['objectType'] for obj in objects 
#             if obj["receptacle"]
#         }

#         for pick in pickupable_object_types:
#             if pick in DEFAULT_COMPATIBLE_RECEPTACLES:
#                 place_sample = [
#                     obj for obj in receptacle_object_types
#                     if obj in DEFAULT_COMPATIBLE_RECEPTACLES[pick]
#                 ]
#             else:
#                 place_sample = receptacle_object_types
#             for place in place_sample:
#                 scene_possible_pick_place_tasks.append((pick, place))
        
#         for possible_task in scene_possible_pick_place_tasks:
#             out[scene].append(
#                 {
#                     "pickup_target": possible_task[0],
#                     "place_target": possible_task[1],
#                 }
#             )
#     return out

def home_service_simple_pick_and_place_datagen_worker(
    input_queue: mp.Queue,
    output_queue: mp.Queue,
):
    env = HomeServiceTHOREnvironment(
        force_cache_reset=True, controller_kwargs={"commit_id": THOR_COMMIT_ID},
    )

    while True:
        try:
            scene, stage, task_specs = input_queue.get(timeout=2)
        except queue.Empty:
            break
        data = generate_home_service_simple_pick_and_place_for_scenes(
            env=env,
            scene=scene,
            task_spec_dicts=task_specs
        )
        output_queue.put((scene, stage, data[scene]))


if __name__ == "__main__":
    nprocesses = max(mp.cpu_count() // 2, 1)
    stage_to_scenes = {
        stage: get_scenes(stage) for stage in ("train", "train_unseen", "val", "test")
    }
    stage_to_scenes["combined"] = get_scenes("all")
    os.makedirs(STARTER_HOME_SERVICE_SIMPLE_PICK_AND_PLACE_DATA_DIR, exist_ok=True)
    stage_to_scene_to_task_spec_dicts = {stage: {} for stage in stage_to_scenes}
    for stage in stage_to_scenes:
        stage_to_scene_to_task_spec_dicts[stage] = HomeServiceTaskSampler.load_rearrange_data_from_path(
            stage=stage, base_dir=STARTER_REARRANGE_DATA_DIR
        )

    env = HomeServiceTHOREnvironment(
        force_cache_reset=True, controller_kwargs={"commit_id": THOR_COMMIT_ID},
    )
    for stage in stage_to_scenes:
        scene_to_pick_and_places = {}
        total_datapoints = 0
        for scene in stage_to_scene_to_task_spec_dicts[stage]:
            task_data = generate_home_service_simple_pick_and_place_for_scenes(
                env=env, scene=scene, task_spec_dicts=stage_to_scene_to_task_spec_dicts[stage][scene]
            )
            scene_to_pick_and_places[scene] = task_data[scene]
        
        compress_pickle.dump(
            obj=scene_to_pick_and_places,
            path=os.path.join(STARTER_HOME_SERVICE_SIMPLE_PICK_AND_PLACE_DATA_DIR, f"{stage}.pkl.gz"),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        

    # import pdb; pdb.set_trace()
    # stage_to_scene_to_pick_and_places = {stage: {} for stage in stage_to_scenes}
    # for stage in stage_to_scenes:
    #     path = os.path.join(STARTER_HOME_SERVICE_SIMPLE_PICK_AND_PLACE_DATA_DIR, f"{stage}.json")
    #     if os.path.exists(path):
    #         # stage_to_scene_to_pick_and_places[stage] = compress_pickle.load(path)
    #         with open(path, "r") as f:
    #             stage_to_scene_to_pick_and_places[stage] = json.load(f)

    # send_queue = mp.Queue()
    # num_scenes_to_run = 0
    # for stage in stage_to_scenes:
    #     for scene in stage_to_scene_to_task_spec_dicts[stage]:
    #         if scene not in stage_to_scene_to_pick_and_places[stage]:
    #             num_scenes_to_run += 1
    #             send_queue.put((scene, stage, stage_to_scene_to_task_spec_dicts[stage][scene]))

    # receive_queue = mp.Queue()
    # processes = []
    # for i in range(nprocesses):
    #     p = mp.Process(
    #         target=home_service_simple_pick_and_place_datagen_worker,
    #         kwargs=dict(
    #             input_queue=send_queue,
    #             output_queue=receive_queue,
    #         ),
    #     )
    #     p.start()
    #     processes.append(p)
    #     time.sleep(0.5)

    # num_received = 0

    # while num_scenes_to_run > num_received:
    #     try:
    #         scene, stage, data = receive_queue.get(timeout=1)
    #         num_received += 1
    #     except queue.Empty:
    #         continue
        
    #     # scene_to_pick_and_places = stage_to_scene_to_pick_and_places[stage]
    #     # if scene not in scene_to_pick_and_places:
    #     #     scene_to_pick_and_places[scene] = []
    #     if scene not in stage_to_scene_to_pick_and_places[stage]:
    #         stage_to_scene_to_pick_and_places[stage][scene] = []

    #     stage_to_scene_to_pick_and_places[stage][scene].extend(data)

    #     with open(os.path.join(STARTER_HOME_SERVICE_SIMPLE_PICK_AND_PLACE_DATA_DIR, f"{stage}.json"), "w") as f:
    #         json.dump(stage_to_scene_to_pick_and_places[stage], f, indent=4)
        
    #     compress_pickle.dump(
    #         obj=stage_to_scene_to_pick_and_places[stage],
    #         path=os.path.join(STARTER_HOME_SERVICE_SIMPLE_PICK_AND_PLACE_DATA_DIR, f"{stage}.pkl.gz"),
    #         protocol=pickle.HIGHEST_PROTOCOL,
    #     )

    # for p in processes:
    #     try:
    #         p.join(timeout=1)
    #     except:
    #         pass




from collections import defaultdict
from typing import Dict, Any
import torch
import os

from ai2thor.platform import CloudRendering
from env.environment import HomeServiceEnvironment
from datagen.datagen_constants import (
    GOOD_4_ROOM_HOUSES,
    STARTER_HOME_SERVICE_DATA_DIR,
    PICKUPABLE_RECEPTACLE_PAIRS,
    PICKUP_OBJECTS_FOR_TEST,
    STAGE_TO_MIN_SCENES,
    STAGE_TO_DEST_NUM_SCENES,
)


# THOR_CONTROLLER_KWARGS = {
#     "snapToGrid": True,
#     "fastActionEmit": True,
#     "renderDepthImage": True,
#     "scene": "Procedural",
#     "x_display": None,
#     "gpu_device": 0,
#     "platform": CloudRendering,
# }

# env = HomeServiceEnvironment(
#     force_cache_reset=True,
#     controller_kwargs=THOR_CONTROLLER_KWARGS,
# )

def gen_metadata(env: HomeServiceEnvironment, save: bool = True, overwrite: bool = False):
    stage_to_pickupable_type_to_scenes = dict()
    stage_to_receptacle_type_to_scenes = dict()
    stage_to_scenes = dict()

    for split, scene_idxs in GOOD_4_ROOM_HOUSES.items():
        pickupable_type_to_scenes = defaultdict(set)
        receptacle_type_to_scenes = defaultdict(set)
        stage_to_scenes[split] = dict()

        for scene_idx in scene_idxs:
            scene = f"{split}_{scene_idx}"
            env.procthor_reset(scene_name=scene)
            if (
                set([room["roomType"] for room in env.current_house["rooms"]])
                != set(['Bathroom', 'Bedroom', 'Kitchen', 'LivingRoom'])
            ):
                continue

            room_type_to_room_id = {room["roomType"]: room['id'] for room in env.current_house["rooms"]}

            room_id_to_pickupable_ids = {
                **{
                    k: list(v)
                    for k, v in env.room_to_pickupable_ids().items()
                }
            }
            room_id_to_static_receptacle_ids = {
                **{
                    k: list(v)
                    for k, v in env.room_to_static_receptacle_ids().items()
                }
            }
            
            room_id_to_pickupable_type_to_ids = {
                room_id: defaultdict(list)
                for room_id in room_id_to_pickupable_ids
            }
            for room_id, pids in room_id_to_pickupable_ids.items():
                for pid in pids:
                    ptype = env.ids_to_objs()[pid]["objectType"]
                    room_id_to_pickupable_type_to_ids[room_id][ptype].append(pid)

            room_id_to_pickupable_type_to_num = {
                room_id: {
                    ptype: len(pids)
                    for ptype, pids in room_id_to_pickupable_type_to_ids[room_id].items()
                }
                for room_id in room_id_to_pickupable_type_to_ids
            }

            room_id_to_receptacle_type_to_ids = {
                room_id: defaultdict(list)
                for room_id in room_id_to_static_receptacle_ids
            }
            for room_id, rids in room_id_to_static_receptacle_ids.items():
                for rid in rids:
                    rtype = env.ids_to_objs()[rid]["objectType"]
                    room_id_to_receptacle_type_to_ids[room_id][rtype].append(rid)

            room_id_to_receptacle_type_to_num = {
                room_id: {
                    rtype: len(rids)
                    for rtype, rids in room_id_to_receptacle_type_to_ids[room_id].items()
                }
                for room_id in room_id_to_receptacle_type_to_ids
            }

            for room_id in room_id_to_pickupable_type_to_ids:
                for ptype in room_id_to_pickupable_type_to_ids[room_id].keys():
                    pickupable_type_to_scenes[ptype].add(scene)
            for room_id in room_id_to_receptacle_type_to_ids:
                for rtype in room_id_to_receptacle_type_to_ids[room_id].keys():
                    receptacle_type_to_scenes[rtype].add(scene)

            scene_info = {
                "room_type_to_room_id": room_type_to_room_id,
                "pickupables": {
                    "room_id_to_obj_ids": room_id_to_pickupable_ids,
                    "room_id_to_obj_types_to_ids": room_id_to_pickupable_type_to_ids,
                    "room_id_to_obj_type_to_num": room_id_to_pickupable_type_to_num,
                },
                "receptacles": {
                    "room_id_to_obj_ids": room_id_to_static_receptacle_ids,
                    "room_id_to_obj_types_to_ids": room_id_to_receptacle_type_to_ids,
                    "room_id_to_obj_type_to_num": room_id_to_receptacle_type_to_num,
                },
            }
            stage_to_scenes[split][scene] = scene_info
        stage_to_pickupable_type_to_scenes[split] = {
            **{
                k: list(v)
                for k, v in pickupable_type_to_scenes.items()
            }
        }
        stage_to_receptacle_type_to_scenes[split] = {
            **{
                k: list(v)
                for k, v in receptacle_type_to_scenes.items()
            }
        }

    metadata = {
        stage: {
            "scenes": stage_to_scenes[stage],
            "pickupables": stage_to_pickupable_type_to_scenes[stage],
            "receptacles": stage_to_receptacle_type_to_scenes[stage],
        }
        for stage in ("train", "val", "test")
    }

    if save:
        save_metadata(
            base_dir=STARTER_HOME_SERVICE_DATA_DIR,
            data=metadata,
            overwrite=overwrite,
            format="both",
        )


def save_metadata(base_dir: str, data: Dict[str, Any], overwrite: bool, format: str = "json"):
    assert format in ("json", "compress_pickle", "both")
    assert os.path.exists(base_dir)

    def save_json(fpath, d):
        import json
        with open(fpath, "w") as f:
            json.dump(d, f)

    def save_compress_pickle(fpath, d):
        import compress_pickle
        compress_pickle.dump(
            obj=d,
            path=fpath,
        )

    if format in ("json", "both"):
        fpath = os.path.join(base_dir, "metadata", "metadata.json")
        if overwrite or not os.path.exists(fpath):
            save_json(fpath, data)
        else:
            print(f"file already exists... {fpath}")
    if format in ("compress_pickle", "both"):
        fpath = os.path.join(base_dir, "metadata", "metadata.pkl.gz")
        if overwrite or not os.path.exists(fpath):
            save_compress_pickle(fpath, data)
        else:
            print(f"file already exists... {fpath}")


def sort_scenes(
    base_dir: str = STARTER_HOME_SERVICE_DATA_DIR,
):
    import compress_pickle
    metadata = compress_pickle.load(
        os.path.join(base_dir, "metadata", f"metadata.pkl.gz")
    )
    stage_to_tasks = {
        stage: [
            f"{stage}_pick_and_place_{pick}_{recep}"
            for pick, recep in PICKUPABLE_RECEPTACLE_PAIRS
            if (
                pick not in (PICKUP_OBJECTS_FOR_TEST if 'train' in stage else [])
                and pick in metadata[stage]["pickupables"]
                and recep in metadata[stage]["receptacles"]
            )
        ]
        for stage in metadata
    }

    stage_to_task_to_scenes = {stage: {} for stage in stage_to_tasks}
    for stage in stage_to_tasks:
        for task in stage_to_tasks[stage]:
            pick, recep = task.split("_")[-2:]
            stage_to_task_to_scenes[stage][task] = []
            for scene in metadata[stage]["scenes"]:
                if not (
                    scene in metadata[stage]["pickupables"][pick]
                    and scene in metadata[stage]["receptacles"][recep]
                ):
                    continue

                num_picks = 0
                num_receps = 0
                for room_id in metadata[stage]["scenes"][scene]["pickupables"]["room_id_to_obj_ids"]:
                    if pick in metadata[stage]["scenes"][scene]["pickupables"]["room_id_to_obj_type_to_num"][room_id]:
                        num_picks += metadata[stage]["scenes"][scene]["pickupables"]["room_id_to_obj_type_to_num"][room_id][pick]

                for room_id in metadata[stage]["scenes"][scene]["receptacles"]["room_id_to_obj_ids"]:
                    if recep in metadata[stage]["scenes"][scene]["receptacles"]["room_id_to_obj_type_to_num"][room_id]:
                        num_receps += metadata[stage]["scenes"][scene]["receptacles"]["room_id_to_obj_type_to_num"][room_id][recep]

                if (num_picks == 1 and num_receps == 1):
                    stage_to_task_to_scenes[stage][task].append(scene)

    stage_to_task_to_num_scenes = {
        stage: {
            task: len(stage_to_task_to_scenes[stage][task])
            for task in tasks
        }
        for stage, tasks in stage_to_task_to_scenes.items()
    }

    stage_to_valid_tasks = {
        stage: [task for task in tasks if stage_to_task_to_num_scenes[stage][task] >= STAGE_TO_MIN_SCENES[stage]]
        for stage, tasks in stage_to_tasks.items()
    }

    stage_to_valid_task_to_scenes = {
        stage: {
            task: stage_to_task_to_scenes[stage][task]
            for task in tasks
        }
        for stage, tasks in stage_to_valid_tasks.items()
    }

    stage_to_valid_task_to_num_scenes = {
        stage: {
            task: len(stage_to_valid_task_to_scenes[stage][task])
            for task in tasks
        }
        for stage, tasks in stage_to_valid_task_to_scenes.items()
    }

    stage_to_valid_task_to_num_scenes_sorted = {
        stage: dict(sorted(stage_to_valid_task_to_num_scenes[stage].items(), key=lambda item: item[1]))
        for stage in stage_to_valid_task_to_num_scenes
    }

    stage_to_scene_to_tasks = {
        stage: {} for stage in stage_to_tasks
    }
    for stage in stage_to_tasks:
        for task, scenes in stage_to_task_to_scenes[stage].items():
            for scene in scenes:
                if scene not in stage_to_scene_to_tasks[stage]:
                    stage_to_scene_to_tasks[stage][scene] = []
                stage_to_scene_to_tasks[stage][scene].append(task)
    
    stage_to_scene_to_num_tasks = {
        stage: {
            scene: len(tasks)
            for scene, tasks in stage_to_scene_to_tasks[stage].items()
        }
        for stage in stage_to_tasks
    }

    stage_to_scene_to_valid_tasks = {
        stage: {
            scene: [
                task
                for task in stage_to_scene_to_tasks[stage][scene]
                if task in stage_to_valid_tasks[stage]
            ]
            for scene in stage_to_scene_to_tasks[stage]
        }
        for stage in stage_to_tasks
    }

    stage_to_scene_to_num_valid_tasks = {
        stage: {
            scene: len(tasks)
            for scene, tasks in stage_to_scene_to_valid_tasks[stage].items()
        }
        for stage in stage_to_tasks
    }

    stage_to_scene_to_num_valid_tasks_sorted = {
        stage: dict(sorted(stage_to_scene_to_num_valid_tasks[stage].items(), key=lambda item: item[1], reverse=True))
        for stage in stage_to_tasks
    }

    stage_to_task_to_scenes_selected = {}
    stage_to_task_to_num_scenes_selected = {}
    stage_to_scene_set_selected = {}
    for stage in stage_to_valid_task_to_num_scenes_sorted:
        scene_set = set()
        task_to_scenes = {}
        for task in stage_to_valid_task_to_num_scenes_sorted[stage]:
            for scene in stage_to_valid_task_to_scenes[stage][task]:
                if scene not in scene_set:
                    scene_set.add(scene)
                    for task in stage_to_scene_to_valid_tasks[stage][scene]:
                        if task not in task_to_scenes:
                            task_to_scenes[task] = set()
                        task_to_scenes[task].add(scene)
            task_to_num_scenes = {
                task: len(task_to_scenes[task])
                for task in task_to_scenes
            }
            task_to_scenes = dict(
                sorted(
                    task_to_scenes.items(),
                    key=lambda item: task_to_num_scenes[item[0]],
                )
            )
            task_to_num_scenes = dict(
                sorted(
                    task_to_num_scenes.items(),
                    key=lambda item: item[1],
                )
            )
            if task_to_num_scenes[list(task_to_scenes.keys())[0]] >= STAGE_TO_DEST_NUM_SCENES[stage]:
                break
        
        stage_to_task_to_scenes_selected[stage] = task_to_scenes
        stage_to_task_to_num_scenes_selected[stage] = task_to_num_scenes
        stage_to_scene_set_selected[stage] = scene_set

    return (
        stage_to_task_to_scenes_selected,
        stage_to_scene_set_selected,
    )
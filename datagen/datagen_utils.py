import random
from collections import defaultdict
from typing import List, Dict, Set, Optional, Any

from ai2thor.controller import Controller
from datagen.datagen_constants import OBJECTS_FOR_TEST, TASK_ORDERS

from env.constants import SCENE_TYPE_TO_SCENES


def get_scenes(stage: str) -> List[str]:
    """Returns a list of iTHOR scene names for each stage."""
    assert stage in {"train", "train_unseen", "val", "valid", "test", "all"}
    assert stage in {"debug", "train", "train_unseen", "val", "valid", "test", "all"}

    if stage == "debug":
        return ["FloorPlan1"]

    # [1-20] for train, [21-25] for val, [26-30] for test
    if stage in ["train", "train_unseen"]:
        scene_nums = range(1, 21)
    elif stage in ["val", "valid"]:
        scene_nums = range(21, 26)
    elif stage == "test":
        scene_nums = range(26, 31)
    elif stage == "all":
        scene_nums = range(1, 31)
    else:
        raise NotImplementedError

    kitchens = [f"FloorPlan{i}" for i in scene_nums]
    living_rooms = [f"FloorPlan{200+i}" for i in scene_nums]
    bedrooms = [f"FloorPlan{300+i}" for i in scene_nums]
    bathrooms = [f"FloorPlan{400+i}" for i in scene_nums]
    return kitchens + living_rooms + bedrooms + bathrooms


def filter_pickupable(
    objects: List[Dict], object_types_to_not_move: Set[str]
) -> List[Dict]:
    """Filters object data only for pickupable objects."""
    return [
        obj
        for obj in objects
        if obj["pickupable"] and not obj["objectType"] in object_types_to_not_move
    ]


def get_random_seeds(max_seed: int = int(1e8)) -> Dict[str, int]:
    # Generate random seeds for each stage

    # Train seen seed
    random.seed(1329328939)
    train_seen_seed = random.randint(0, max_seed - 1)

    # Train unseen seed
    random.seed(709384928)
    train_unseen_seed = random.randint(0, max_seed - 1)

    # Test seen seed
    random.seed(3348958620)
    test_seen_seed = random.randint(0, max_seed - 1)

    # Test unseen seed
    random.seed(289123396)
    test_unseen_seed = random.randint(0, max_seed - 1)

    # Debug seed
    random.seed(239084231)
    debug_seed = random.randint(0, max_seed - 1)

    return {
        "train_seen": train_seen_seed,
        "train_unseen": train_unseen_seed,
        "test_seen": test_seen_seed,
        "test_unseen": test_unseen_seed,
        "debug": debug_seed,
    }


def open_objs(
    objects_to_open: List[Dict[str, Any]], controller: Controller
) -> Dict[str, Optional[float]]:
    """Opens up the chosen pickupable objects if they're openable."""
    out: Dict[str, Optional[float]] = defaultdict(lambda: None)
    for obj in objects_to_open:
        last_openness = obj["openness"]
        new_openness = last_openness
        while abs(last_openness - new_openness) <= 0.2:
            new_openness = random.random()

        controller.step(
            "OpenObject",
            objectId=obj["objectId"],
            openness=new_openness,
            forceAction=True,
        )
        out[obj["name"]] = new_openness
    return out


def get_object_ids_to_not_move_from_object_types(
    controller: Controller, object_types: Set[str]
) -> List[str]:
    object_types = set(object_types)
    return [
        o["objectId"]
        for o in controller.last_event.metadata["objects"]
        if o["objectType"] in object_types
    ]


# def remove_objects_until_all_have_identical_meshes(controller: Controller):
#     obj_type_to_obj_list = defaultdict(lambda: [])
#     for obj in controller.last_event.metadata["objects"]:
#         obj_type_to_obj_list[obj["objectType"]].append(obj)

#     for obj_type in OBJECT_TYPES_THAT_CAN_HAVE_IDENTICAL_MESHES:
#         objs_of_type = list(
#             sorted(obj_type_to_obj_list[obj_type], key=lambda x: x["name"])
#         )
#         random.shuffle(objs_of_type)
#         objs_to_remove = objs_of_type[:-1]
#         for obj_to_remove in objs_to_remove:
#             obj_to_remove_name = obj_to_remove["name"]
#             obj_id_to_remove = next(
#                 obj["objectId"]
#                 for obj in controller.last_event.metadata["objects"]
#                 if obj["name"] == obj_to_remove_name
#             )
#             controller.step("RemoveFromScene", objectId=obj_id_to_remove)
#             if not controller.last_event.metadata["lastActionSuccess"]:
#                 return False
#     return True

def find_object_by_type(objects: List[Dict], object_type: str):
    return [
        obj
        for obj in objects
        if obj["objectType"] == object_type
    ]

def scene_from_type_idx(scene_type: str, scene_idx: int):
    return SCENE_TYPE_TO_SCENES[scene_type][scene_idx - 1]

def get_task_keys(stage: str) -> List[str]:
    assert stage in {"train_seen", "train_unseen", "test_seen", "test_unseen", "all"}
    stage = stage.split("_")[0]

    task_keys = []
    for task_order in TASK_ORDERS:
        pickup_object_types = task_order["pickupObjectTypes"]
        for pickup_object_type in pickup_object_types:
            task_key = (
                f'Pick_{pickup_object_type}_On_{task_order["startReceptacleType"]}_And_Place_{task_order["targetReceptacleType"]}'
                if task_order["targetReceptacleType"] != "User" else
                f'Bring_Me_{pickup_object_type}_On_{task_order["startReceptacleType"]}'
            )
            if stage == "test" and pickup_object_type in OBJECTS_FOR_TEST:
                task_keys.append(task_key)
            elif stage == "train" and pickup_object_type not in OBJECTS_FOR_TEST:
                task_keys.append(task_key)
            elif stage == "all":
                task_keys.append(task_key)
            
    return task_keys

def get_scene_inds(stage: str, seen: bool = None) -> List[int]:
    assert stage in {"train", "test", "train_seen", "train_unseen", "test_seen", "test_unseen", "all"}
    if stage in {"train", "test"}:
        assert seen is not None
    elif stage == "all":
        return range(1, 31)
    else:
        seen = True if stage.split("_")[-1] == "seen" else False
    
    if seen:
        return range(1, 21)
    else:
        return range(21, 31)

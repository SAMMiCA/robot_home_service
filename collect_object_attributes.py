from typing import Dict, Any, Sequence
import json
from env.constants import OBJECT_TYPES_WITH_PROPERTIES, REARRANGE_SIM_OBJECTS
from env.tasks import HomeServiceTaskSampler
from experiments.home_service_base import HomeServiceBaseExperimentConfig

task_sampler_params = HomeServiceBaseExperimentConfig.stagewise_task_sampler_args(
    stage="combined", process_ind=0, total_processes=1, headless=False, allowed_inds_subset=tuple(range(1)),
)

task_sampler: HomeServiceTaskSampler = HomeServiceBaseExperimentConfig.make_sampler_fn(
    **task_sampler_params,
    force_cache_reset=True,
    runtime_sample=False,
    repeats_before_scene_change=1,
    epochs=1,
)

scene_nums = range(1, 31)
kitchens = [f"FloorPlan{i}" for i in scene_nums]
living_rooms = [f"FloorPlan{200+i}" for i in scene_nums]
bedrooms = [f"FloorPlan{300+i}" for i in scene_nums]
bathrooms = [f"FloorPlan{400+i}" for i in scene_nums]
rooms = {
    "kitchen": kitchens,
    "livingRoom": living_rooms,
    "bedroom": bedrooms,
    "bathroom": bathrooms,
}

def duplicated_object(obj: Dict[str, Any], object_list: Sequence[Dict[str, Any]]):
    if len(object_list) == 0:
        return False
    else:
        for obj_ in object_list:
            duplicated = True

            for key in ["objectType", "ObjectTemperature", "isCooked", "isSliced", "isToggled", "isBroken", "isFilledWithLiquid", "isDirty", "isUsedUp"]:
                if obj[key] != obj_[key]:
                    duplicated = False

            # obj_parent_receptacles = [] if obj["parentReceptacles"] is None else obj["parentReceptacles"]
            # obj_parent_receptacles_ = [] if obj_["parentReceptacles"] is None else obj_["parentReceptacles"]

            # if len(obj_parent_receptacles) != len(obj_parent_receptacles_):
            #     return False
            # else:
            #     for receptacle, receptacle_ in zip(obj_parent_receptacles, obj_parent_receptacles_):
            #         if receptacle.split("|")[0] != receptacle_.split("|")[0]:
            #             return False

            # obj_receptacle_objects = [] if obj["receptacleObjectIds"] is None else obj["receptacleObjectIds"]
            # obj_receptacle_objects_ = [] if obj_["receptacleObjectIds"] is None else obj_["receptacleObjectIds"]

            # if len(obj_receptacle_objects) != len(obj_receptacle_objects_):
            #     return False
            # else:
            #     for receptacle, receptacle_ in zip(obj_receptacle_objects, obj_receptacle_objects_):
            #         if receptacle.split("|")[0] != receptacle_.split("|")[0]:
            #             return False

            obj_salient_materials = [] if obj["salientMaterials"] is None else obj["salientMaterials"]
            obj_salient_materials_ = [] if obj_["salientMaterials"] is None else obj_["salientMaterials"]

            if len(obj_salient_materials) != len(obj_salient_materials_):
                duplicated = False
            else:
                if set(obj_salient_materials) != set(obj_salient_materials_):
                    duplicated = False
            
            if duplicated:
                break
        
        return duplicated

save_dict = []
for i_task in range(task_sampler.total_unique):
    task = task_sampler.next_task()
    objects = task_sampler.env.last_event.metadata["objects"]

    for obj in objects:
        if obj["objectType"] in REARRANGE_SIM_OBJECTS and (
            any(OBJECT_TYPES_WITH_PROPERTIES[obj["objectType"]].values())
        ) and not (
            duplicated_object(obj, save_dict)
        ):
            obj["scene"] = task_sampler.current_task_spec.scene
            for room in rooms.keys():
                if obj["scene"] in rooms[room]:
                    obj["roomType"] = room
                    break
            del_keys = ["position", "rotation", "visible", "axisAlignedBoundingBox", "objectOrientedBoundingBox"]
            for key in del_keys:
                del obj[key]            
            save_dict.append(obj)

    # duplicated = [obj for i, obj in enumerate(save_dict) if duplicated_object(obj, save_dict[:i] + save_dict[i+1:])]
    # while duplicated:
    #     t_obj = duplicated[0]
    #     duplicated_inds = [i for i, obj in enumerate(save_dict) if obj["objectType"] == t_obj["objectType"]]
    #     for ind in duplicated_inds[1:]:
    #         del save_dict[ind]
        
    #     duplicated = [obj for i, obj in enumerate(save_dict) if duplicated_object(obj, save_dict[:i] + save_dict[i+1:])]

print(f"total number of objects: {len(save_dict)}")
with open("objects.json", "w") as fp:
    json.dump(save_dict, fp, indent=4)
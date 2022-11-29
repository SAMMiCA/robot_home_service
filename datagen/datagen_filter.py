import os
import json
import compress_pickle
from datagen.datagen_constants import STARTER_HOME_SERVICE_DATA_DIR, STAGE_TO_DEST_NUM_SCENES


split = ("train", "val", "test")
data = {
    stage: compress_pickle.load(
        os.path.join(STARTER_HOME_SERVICE_DATA_DIR, f'rawdata/{stage}.pkl.gz')
    )
    for stage in split
}

filtered_data = dict(
    not_has_none=dict(),
    not_all_none=dict(),
)
for stage in data:
    has_none_task = []
    all_none_task = []
    num_houses = STAGE_TO_DEST_NUM_SCENES[stage]
    num_scenes = 5 if stage == "train" else 2

    for task, list_spec_dict in data[stage].items():
        has_none = not all(
            list_spec_dict[i][j] is not None
            for i in range(num_houses)
            for j in range(num_scenes)
        )
        all_none = all(
            list_spec_dict[i][j] is None
            for i in range(num_houses)
            for j in range(num_scenes)
        )
        if has_none:
            has_none_task.append(task)
        if all_none:
            all_none_task.append(task)
    
    filtered_data["not_has_none"][stage] = {
        k: v for k, v in data[stage].items()
        if k not in has_none_task
    }
    filtered_data["not_all_none"][stage] = {
        k: v for k, v in data[stage].items()
        if k not in all_none_task
    }

for stage in data:
    compress_pickle.dump(
        obj=filtered_data["not_has_none"][stage],
        path=os.path.join(STARTER_HOME_SERVICE_DATA_DIR, f"{stage}.pkl.gz"),
    )
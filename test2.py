import random
from typing import List
import matplotlib.pyplot as plt
from PIL import Image
import json
from ai2thor.controller import Controller
from allenact_plugins.ithor_plugin.ithor_util import include_object_data

start_room = "FloorPlan2"
# c = Controller(quality="Ultra", local_executable_path="unity/robot_home_service.x86_64", scene=start_room)
# with open("/media/yhkim/HDD/ALFRED_Dataset/json_2.1.0/train/pick_and_place_simple-Box-None-Dresser-224/trial_T20190907_163906_753897/traj_data.json", "r") as f:
#     data = json.load(f)

# c = Controller(quality="Ultra", commit_id="f46d5ec42b65fdae9d9a48db2b4fb6d25afbd1fe", scene=data['scene']['floor_plan'])
c = Controller(quality="Ultra", commit_id="f46d5ec42b65fdae9d9a48db2b4fb6d25afbd1fe", scene=start_room)
# c.step(dict(action='SetObjectPoses', objectPoses=data['scene']['object_poses']))
# obj_poses = [
#     {
#         "objectName": "Mug_8903bf3b",
#         "position": {
#             "x": -0.431458145,
#             "y": 0.6181135,
#             "z": 0.5991447
#         },
#         "rotation": {
#             "x": 0.0,
#             "y": 180.00032,
#             "z": 0.0
#         }
#     },
#     {
#         "objectName": "Mug_8903bf3b",
#         "position": {
#             "x": -0.7414144,
#             "y": 0.616495252,
#             "z": 1.06685877
#         },
#         "rotation": {
#             "x": 0.0,
#             "y": 180.00032,
#             "z": 0.0
#         }
#     },
#     {
#         "objectName": "Mug_123",
#         "position": {
#             "x": -0.58643496,
#             "y": 0.6181135,
#             "z": 1.06685793
#         },
#         "rotation": {
#             "x": 0.0,
#             "y": 180.00032,
#             "z": 0.0
#         }
#     }
# ]
import pdb; pdb.set_trace()
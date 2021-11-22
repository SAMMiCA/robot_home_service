import random
import os
from typing import List
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import json
from ai2thor.controller import Controller
from allenact_plugins.ithor_plugin.ithor_util import include_object_data

start_room = "FloorPlan2"
command_to_action_str = {
    1: "MoveAhead",
    2: "RotateRight",
    3: "RotateLeft",
    4: "NextRoom",
    5: "SavePoint",
}

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.ion()
# plt.show()

doors_points = dict()
json_f = "door_points.json"
if os.path.exists(json_f):
    with open(json_f, "r") as f:
        doors_points = json.load(f)

c = Controller(quality="Ultra", commit_id="f46d5ec42b65fdae9d9a48db2b4fb6d25afbd1fe", scene=start_room)

for j in range(1, 31):
    for i in (0, 2, 3, 4):
        room = f"FloorPlan{i * 100 + j}"
        print(f"{room}...")
        if room in doors_points:
            print(f'already done... skip...')
            continue
        c.reset(scene=room)
        doors_points[room] = []
        rgb = c.last_event.frame
        # ax.imshow(rgb)
        # plt.draw()
        # plt.pause(0.001)

        input_command = int(input("input command: "))
        while input_command != -1:
            action_str = command_to_action_str[input_command]
            if input_command in (1, 2, 3):
                c.step(action=action_str)
            elif input_command == 4:
                with open('door_points.json', 'w') as f:
                    json.dump(doors_points, f)
                break
            elif input_command == 5:
                agent_position = c.last_event.metadata['agent']['position']
                agent_rotation = c.last_event.metadata['agent']['rotation']
                door_point = (agent_position['x'], agent_position['z'], agent_rotation['y'])
                print(f"Saving door_point: {door_point}")
                doors_points[room].append(door_point)

            rgb = c.last_event.frame
            # ax.imshow(rgb)
            # plt.draw()
            # plt.pause(0.001)

            input_command = int(input("input command: "))
    
import pdb; pdb.set_trace()
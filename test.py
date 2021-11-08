import random
from typing import List
import matplotlib.pyplot as plt
from PIL import Image
from ai2thor.controller import Controller
from allenact_plugins.ithor_plugin.ithor_util import include_object_data

def pickup_apple(c):
    with include_object_data(c):
        objs = c.last_event.metadata['objects']
        apple_id = next(
            (
                o['objectId'] for o in objs
                if o['objectType'] == 'Apple' and o['visible']
            ), None
        )
        if apple_id is not None:
            c.step(
                action="PickupObject", 
                objectId=apple_id
            )

def reset_scene(c, home: List[str], held_apple: bool, step: int):
    scene_next = random.choice(home)
    c.reset(scene=scene_next)
    if held_apple:
        c.step("CreateObject", objectType="Apple")
    rgb = c.last_event.frame
    for _ in range(8):
        c.step("RotateRight", degrees=45)

    return scene_next
    

def return_scene(c, home: List[str], scene_before: str, scene_current: str, held_apple: bool):
    c.reset(scene=scene_before)
    if held_apple:
        c.step("CreateObject", objectType="Apple")

    home = home.remove(scene_current)
    return home, scene_before, scene_current

home_dicts = {f"Home{k}": [f"FloorPlan{k + i * 100}" for i in [0, 2, 3, 4]] for k in range(1, 31)}

task_home = home_dicts["Home1"]
start_room = random.choice(task_home[1:])
c = Controller(quality="Ultra", local_executable_path="unity/robot_home_service.x86_64", scene=start_room)

command_to_action_str = {
    1: "MoveAhead",
    2: "RotateRight",
    3: "RotateLeft",
    4: "PikcupApple",
    5: "ResetScene",
    6: "ReturnScene",
}

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()
plt.show()

scene_current = start_room
scene_before = None
input_command = int(input("input_command: "))
step = 0
while input_command != -1:
    action_str = command_to_action_str[input_command]
    with include_object_data(c):
        metadata = c.last_event.metadata
        if len(metadata["inventoryObjects"]) == 0:
            held_apple = False
        elif len(metadata["inventoryObjects"]) == 1:
            held_apple = metadata["inventoryObjects"][0]["objectType"] == "Apple"
    
    if input_command in [1, 2, 3]:
        c.step(action=action_str)
    elif input_command == 4:
        pickup_apple(c)
    elif input_command == 5:
        scene_before = scene_current
        task_home = [scene for scene in task_home if scene != scene_current]
        scene_next = random.choice(task_home)
        c.reset(scene=scene_next)
        if held_apple:
            c.step("CreateObject", objectType="Apple")
        for _ in range(8):
            rgb = c.last_event.frame
            im = Image.fromarray(rgb)
            im.save(f'test/{step}.png')
            ax.imshow(rgb)
            plt.draw()
            plt.pause(0.001)
            print(c.last_event)
            step += 1
            c.step("RotateRight", degrees=45)
        scene_current = scene_next

    elif input_command == 6:
        c.reset(scene=scene_before)
        if held_apple:
            c.step("CreateObject", objectType="Apple")

        task_home = task_home.remove(scene_current)
        scene_before, scene_current = scene_current, scene_before

    rgb = c.last_event.frame
    im = Image.fromarray(rgb)
    im.save(f'test/{step}.png')
    ax.imshow(rgb)
    plt.draw()
    plt.pause(0.001)
    print(c.last_event)
    step += 1
    input_command = int(input("input_command: "))

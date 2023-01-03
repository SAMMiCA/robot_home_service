import copy
import os
import random
import numpy as np
import time

from env.tasks import HomeServiceTaskSampler
from env.environment import HomeServiceTaskSpec, HomeServiceEnvironment
from allenact.utils.system import init_logging


def test_expert_base(
    stage: str = "train",
    subtask: bool = True,
    skip_between_tasks: int = 0,
    random_action_prob: float = 0.0,
    explore: bool = True,
    include_other_move_actions: bool = True,
    save_result: bool = True,
    resume: bool = False,
    verbose: bool = True,
):
    if subtask:
        from experiments.home_service_il_clip_dagger_subtask import HomeServiceSubtaskClipRN50DaggerDistributedConfig as Config
    else:
        from experiments.home_service_il_clip_dagger import HomeServiceClipRN50DaggerDistributedConfig as Config
    config = Config(
        expert_exploration_enabled=explore,
        include_other_move_actions=include_other_move_actions
    )
    task_sampler_params = config.stagewise_task_sampler_args(
        stage=stage,
        process_ind=0,
        total_processes=1,
        devices=[0, ],
        # allowed_pickup_objs=["Pot"],
        # allowed_target_receps=["Fridge"],
    )
    task_sampler: HomeServiceTaskSampler = config.make_sampler_fn(
        **task_sampler_params,
        force_cache_reset=True,
        repeats_per_task=1,
        epochs=1,
    )

    run_tasks = 0
    successful_tasks = 0
    task_lengths = []
    if save_result:
        tasks = []
        task_ids = []
        task_houses = []
        task_house_scene_i = []
        actions_taken = []
        successes = []
        action_taken_successes = []

        target_object_ids = []
        target_receptacle_ids = []
        target_object_room_ids = []
        target_receptacle_room_ids = []

        target_objects = []
        target_receptacles = []
        target_object_rooms = []
        target_receptacle_rooms = []

        expert_actions = []
        if subtask:
            expert_subtasks = []
    
    if resume:
        import json
        if not os.path.exists(f'test_subtask_expert_{stage}.json'):
            raise FileNotFoundError

        with open(f'test_subtask_expert_{stage}.json', 'r') as fp:
            loaded_result = json.load(fp)
        
        run_tasks = loaded_result["run_tasks"]
        successful_tasks = loaded_result["successful_tasks"]
        task_lengths.extend(
            [
                len(action_taken) for action_taken
                in loaded_result["actions_taken"]
            ]
        )
        tasks.extend(loaded_result["tasks"])
        task_ids.extend(loaded_result["task_ids"])
        task_houses.extend(loaded_result["task_houses"])
        task_house_scene_i.extend(loaded_result["task_house_scene_i"])
        actions_taken.extend(loaded_result["actions_taken"])
        successes.extend(loaded_result["successes"])
        action_taken_successes.extend(loaded_result["action_taken_successes"])

        target_object_ids.extend(loaded_result["target_object_ids"])
        target_receptacle_ids.extend(loaded_result["target_receptacle_ids"])
        target_object_room_ids.extend(loaded_result["target_object_room_ids"])
        target_receptacle_room_ids.extend(loaded_result["target_receptacle_room_ids"])

        target_objects.extend(loaded_result["target_objects"])
        target_receptacles.extend(loaded_result["target_receptacles"])
        target_object_rooms.extend(loaded_result["target_object_rooms"])
        target_receptacle_rooms.extend(loaded_result["target_receptacle_rooms"])

        expert_actions.extend(loaded_result["expert_actions"])
        if subtask:
            expert_subtasks.extend(loaded_result["expert_subtasks"])

        done_task_ids = copy.deepcopy(loaded_result["task_ids"])
        del loaded_result

    if explore:
        arrived_target_recep_room_for_scan = False
        arrived_target_object_room_for_scan = False
        found_target_recep_for_scan = False
        found_target_object_for_scan = False
    picked_up_target_object = False
    put_down_object_on_recep = False

    print("TEST EXPERT PERFORMANCE")
    print(f"    STAGE: {stage}, EXPLORE: {explore}, NUM_TASKS: {task_sampler.length}")

    start_time = time.time()
    total_length = task_sampler.length
    force_task_ids = {
        # 'val__pick_and_place_WineBottle_CounterTop__3__1',
        # 'val__pick_and_place_TissueBox_TVStand__4__0',
    }

    while task_sampler.length > 0:
        print(f"======================== [ {run_tasks+1}-th task ] ========================")
        if resume:
            task_spec: HomeServiceTaskSpec = next(task_sampler.task_spec_iterator)
            while task_spec.unique_id in done_task_ids:
                task_spec = next(task_sampler.task_spec_iterator)
            task = task_sampler.next_task(task_spec)
        elif force_task_ids is not None and len(force_task_ids) > 0:
            task_spec = next(task_sampler.task_spec_iterator)
            while task_spec.unique_id not in force_task_ids:
                task_spec = next(task_sampler.task_spec_iterator)
            task = task_sampler.next_task(task_spec)
        else:
            task = task_sampler.next_task()
        if task is None:
            break

        print(f"TASK UNIQUE ID: {task.env.current_task_spec.unique_id}")
        obs = task.get_observations()
        controller = task_sampler.env.controller
        frames = [controller.last_event.frame]
        while not task.is_done():
            if random.random() < random_action_prob or not bool(
                obs["expert_action"][1]
            ):
                assert task.action_names()[0] == "done"
                if random.random() < 0.5:
                    action_to_take = next(
                        it
                        for it, action in enumerate(task.action_names())
                        if "ove" in action and "head" in action
                    ) # MoveAhead
                else:
                    action_to_take = random.randint(0, len(task.action_names()) - 1)
            else:
                action_to_take = obs["expert_action"][0]
            
            # print(f"task step: {task.num_steps_taken()} | action_to_take: {task.action_names()[action_to_take]} ({action_to_take})")
            if verbose:
                if explore:
                    if (
                        task.env.current_room == task.env.target_recep_room_id
                        and not arrived_target_recep_room_for_scan
                    ):
                        arrived_target_recep_room_for_scan = True
                        print(f"TARGET RECEP ROOM ARRIVED!")

                    if (
                        task.env.current_room == task.env.target_object_room_id
                        and arrived_target_recep_room_for_scan
                        and not arrived_target_object_room_for_scan
                    ):
                        arrived_target_object_room_for_scan = True
                        print(f"TARGET OBJECT ROOM ARRIVED AFTER FINDING TARGET RECEP!!!!!!!")

                    if (
                        task.env.target_recep_id in task.action_expert.visited_recep_ids_per_room[task.env.target_recep_room_id]
                        and not found_target_recep_for_scan
                    ):
                        found_target_recep_for_scan = True
                        print(f'TARGET RECEP Found!!')

                    if (
                        task.env.target_object_id in task.action_expert.scanned_objects
                        and found_target_recep_for_scan
                        and not found_target_object_for_scan
                    ):
                        found_target_object_for_scan = True
                        print(f"TARGET OBJECT FOUND!@#!@#!@#!#@!#")

                if (
                    task.env.held_object is not None
                    and task.env.held_object["objectId"] == task.env.target_object_id
                    and not picked_up_target_object
                ):
                    picked_up_target_object = True
                    print(f"PICKED UP TARGET OBJECT ' 3')b")

                target_recep = next(
                    o for o in task.env.objects() if o["objectId"] == task.env.target_recep_id
                )
                if (
                    task.env.held_object is None
                    and picked_up_target_object
                    and not put_down_object_on_recep
                    and task.env.target_object_id in target_recep["receptacleObjectIds"]
                ):
                    put_down_object_on_recep = True
                    print(f"PUT DOWN HELD TARGET OBJECT ON TARGET RECEPTACLE")

            step_result = task.step(action_to_take)
            obs = step_result.observation
            controller.step("Pass") # Why?

            frames.append(controller.last_event.frame)
        
        run_tasks += 1
        if explore:
            arrived_target_recep_room_for_scan = False
            arrived_target_object_room_for_scan = False
            found_target_recep_for_scan = False
            found_target_object_for_scan = False
        picked_up_target_object = False
        put_down_object_on_recep = False

        metrics = task.metrics()
        task_lengths.append(len(task.action_expert.expert_action_list))
        if metrics["success"] == 1.0:
            successful_tasks += 1
            print("Expert Success")
            if verbose:
                _, goal_poses, cur_poses = task.env.poses
                for cp in cur_poses:
                    if cp["objectId"] == task.env.target_object_id:
                        print(f"RESULT {cp['objectId']} {cp['position']} {cp['parentReceptacles']}")
                    if cp["objectId"] == task.env.target_recep_id:
                        print(f"TARGET RECEPTACLE: {cp['objectId']} {cp['position']} {cp['receptacleObjectIds']}")
        else:
            print("Expert Failed")
            if verbose:
                print(f"Failed task: {task.env.current_task_spec.unique_id}")
                _, goal_poses, cur_poses = task.env.poses
                assert len(goal_poses) == len(cur_poses)
                for gp, cp in zip(goal_poses, cur_poses):
                    if (
                        not gp["broken"]
                        and not cp["broken"]
                        and not HomeServiceEnvironment.are_poses_equal(gp, cp)
                    ):
                        print(
                            f"GOAL {gp['objectId']} {gp['position']} {gp['parentReceptacles']}"
                        )
                        print(
                            f"RESULT {cp['objectId']} {cp['position']} {cp['parentReceptacles']}"
                        )
                    elif cp["broken"] and not gp["broken"]:
                        print(f"broken {gp['type']}")
                
            # import pdb; pdb.set_trace(header="Episode Failed")
        
        # import pdb; pdb.set_trace(header="Printed the RESULT")

        if save_result:
            tasks.append(task.env.current_task_spec.task)
            task_ids.append(metrics['task_info']['unique_id'])
            task_houses.append(metrics['task_info']['scene'])
            task_house_scene_i.append(task.env.current_task_spec.metrics['scene_i'])
            actions_taken.append(metrics['task_info']['actions'])
            action_taken_successes.append(metrics['task_info']['action_successes'])
            successes.append(bool(metrics["success"] == 1.0))
            target_object_ids.append(metrics['task_info']['target_object'])
            target_objects.append(task.env.current_task_spec.pickup_object)
            target_receptacle_ids.append(metrics['task_info']['target_receptacle'])
            target_receptacles.append(task.env.current_task_spec.target_receptacle)
            target_object_room_ids.append(task.env.target_object_room_id)
            target_receptacle_room_ids.append(task.env.target_recep_room_id)
            target_object_rooms.append(task.env.get_room_type(task.env.target_object_room_id))
            target_receptacle_rooms.append(task.env.get_room_type(task.env.target_recep_room_id))
            expert_actions.append(metrics["expert_actions"])
            if subtask:
                expert_subtasks.append(metrics["expert_subtasks"])
        print(
            f"Ran tasks {run_tasks} Success rate: {successful_tasks / run_tasks * 100:.2f}%"
            f" length {np.mean(task_lengths):.2f} "
        )
    
        if save_result and time.time() - start_time > 60 * 30:
            import json
            result_dict = {
                "run_tasks": run_tasks,
                "successful_tasks": successful_tasks,
                "success_rate": (successful_tasks / run_tasks * 100),
                "stage": stage,
                "explore": explore,
                "random_action_prob": random_action_prob,
                "include_other_move_actions": include_other_move_actions,
                "tasks": tasks,
                "task_ids": task_ids,
                "task_houses": task_houses,
                "task_house_scene_i": task_house_scene_i,
                "actions_taken": actions_taken,
                "action_taken_successes": action_taken_successes,
                "successes": successes,
                "target_object_ids": target_object_ids,
                "target_objects": target_objects,
                "target_receptacle_ids": target_receptacle_ids,
                "target_receptacles": target_receptacles,
                "target_object_room_ids": target_object_room_ids,
                "target_receptacle_room_ids": target_receptacle_room_ids,
                "target_object_rooms": target_object_rooms,
                "target_receptacle_rooms": target_receptacle_rooms,
                "expert_actions": expert_actions
            }
            if subtask:
                result_dict["expert_subtasks"] = expert_subtasks
            
            print(f"SAVING RESULTS...")
            print(
                f"Ran tasks {run_tasks}/{total_length} Success rate: {successful_tasks / run_tasks * 100:.2f}%"
                f" length {np.mean(task_lengths):.2f} Remaining tasks {task_sampler.length}..."
            )
            with open(f'test_subtask_expert_{stage}.json', 'w') as fp:
                json.dump(result_dict, fp, sort_keys=False, indent=4)

            start_time = time.time()
    
    print(f"======================== [ ALL TASKS DONE ] ========================")
    if save_result:
        import json
        result_dict = {
            "run_tasks": run_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": (successful_tasks / run_tasks * 100),
            "stage": stage,
            "explore": explore,
            "random_action_prob": random_action_prob,
            "include_other_move_actions": include_other_move_actions,
            "tasks": tasks,
            "task_ids": task_ids,
            "task_houses": task_houses,
            "task_house_scene_i": task_house_scene_i,
            "actions_taken": actions_taken,
            "action_taken_successes": action_taken_successes,
            "successes": successes,
            "target_object_ids": target_object_ids,
            "target_objects": target_objects,
            "target_receptacle_ids": target_receptacle_ids,
            "target_receptacles": target_receptacles,
            "target_object_room_ids": target_object_room_ids,
            "target_receptacle_room_ids": target_receptacle_room_ids,
            "target_object_rooms": target_object_rooms,
            "target_receptacle_rooms": target_receptacle_rooms,
            "expert_actions": expert_actions,
        }
        if subtask:
            result_dict["expert_subtasks"] = expert_subtasks
        
        print(f"SAVING RESULTS...")
        print(
            f"Ran tasks {run_tasks}/{total_length} Success rate: {successful_tasks / run_tasks * 100:.2f}%"
            f" length {np.mean(task_lengths):.2f}"
        )

        with open(f'test_subtask_expert_{stage}.json', 'w') as fp:
            json.dump(result_dict, fp, sort_keys=False, indent=4)

        start_time = time.time()


init_logging("info")
test_expert_base(
    stage="test",
    subtask=True,
    skip_between_tasks=0,
    random_action_prob=0.0,
    explore=True,
    include_other_move_actions=True,
    save_result=True,
    resume=False,
    verbose=True,
)



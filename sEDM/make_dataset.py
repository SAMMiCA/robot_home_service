import csv
import pickle
import random

data = {}
unique_goals = []
with open('data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    data_num = 0
    for i, row in enumerate(csv_reader):
        if i != 0:
            start_room = row[0].split(",")
            goal_room = row[1].split(",")
            goal_obj = row[2].split(",")
            goal_location = row[3].split(",")
            obj_recep = row[4].split(",")
            if len(goal_obj) > 1:
                for obj in goal_obj:
                    goal = "pick {} place {}".format(obj, goal_location[0])
                    if goal in unique_goals:
                        continue
                    else:
                        data["data{}".format(data_num)] = {}
                        data["data{}".format(data_num)]['goal'] = ["pick {} place {}".format(obj, goal_location[0])]
                        data["data{}".format(data_num)]['objects'] = [obj, goal_location[0]]
                        data["data{}".format(data_num)]['instruction'] = ["navigate to {}".format(obj),
                                                                          "pick up the {}".format(obj),
                                                                          "navigate to {}".format(goal_location[0]),
                                                                          "put {} to {}".format(obj, goal_location[0])]
                        data_num += 1
                        unique_goals.append(goal)
            else:
                goal = "pick {} place {}".format(goal_obj[0], goal_location[0])
                if goal in unique_goals:
                    continue
                else:
                    data["data{}".format(data_num)] = {}
                    data["data{}".format(data_num)]['goal'] = ["pick {} place {}".format(goal_obj[0], goal_location[0])]
                    data["data{}".format(data_num)]['objects'] = [goal_obj[0], goal_location[0]]
                    data["data{}".format(data_num)]['instruction'] = ["navigate to {}".format(goal_obj[0]),
                                                                      "pick up the {}".format(goal_obj[0]),
                                                                      "navigate to {}".format(goal_location[0]),
                                                                      "put {} to {}".format(goal_obj[0], goal_location[0])]
                    data_num += 1
                    unique_goals.append(goal)

        else:
            continue

    # test_random = random.sample(range(0, 156), 30)
    # train_random = list(set(range(0, 156)) - set(test_random))
    first = [*range(1,32,2)]
    second = [*range(42,57,3)]
    third = [*range(59,83,2)]
    fourth = [*range(128,138,3)]
    # test = first + second + third + fourth + [139, 141, 143]
    # train = list(set(range(0, 156)) - set(test))
    test = [*range(1,22,2),25,28,*range(31,46,2),51,55,61,62,64]
    train = list(set(range(0, 65)) - set(test))
    train_unique_data = {}
    test_unique_data = {}
    for idx, num in enumerate(test):
        test_unique_data["data{}".format(idx)] = data["data{}".format(num)]
    for idx, num in enumerate(train):
        train_unique_data["data{}".format(idx)] = data["data{}".format(num)]
    train_file = open("train_unique_data.pkl", "wb")
    test_file = open("test_unique_data.pkl", "wb")
    pickle.dump(train_unique_data, train_file)
    pickle.dump(test_unique_data, test_file)
import os
import json
import compress_pickle
from collections import defaultdict
from env.constants import DEFAULT_COMPATIBLE_RECEPTACLES, OBJECT_TYPES_WITH_PROPERTIES, STARTER_HOME_SERVICE_DATA_DIR

PICKUPABLES_TO_RECEPTACLES = {
    pick: [
        recep
        for recep in receps if (
            recep in OBJECT_TYPES_WITH_PROPERTIES
            and not OBJECT_TYPES_WITH_PROPERTIES[recep]["pickupable"]
        )
    ]
    for pick, receps in DEFAULT_COMPATIBLE_RECEPTACLES.items()
    if "Sliced" not in pick
}
PICKUPABLE_RECEPTACLE_PAIRS = [
    (pick, recep)
    for pick, receps in PICKUPABLES_TO_RECEPTACLES.items()
    for recep in receps
]
RECEPTACLES_TO_PICKUPABLES = defaultdict(list)
for pair in PICKUPABLE_RECEPTACLE_PAIRS:
    pick, recep = pair
    RECEPTACLES_TO_PICKUPABLES[recep].append(pick)
RECEPTACLES_TO_PICKUPABLES = dict(RECEPTACLES_TO_PICKUPABLES)

PICKUP_OBJECTS_FOR_TRAIN = ["Bowl", "Cup", "Kettle", "Pan", "Tomato", "Bottle", "Box", "Laptop", "RemoteControl", "Newspaper", "TissueBox", "ToiletPaper"]
PICKUP_OBJECTS_FOR_TEST = ["Plate", "Mug", "Pot", "Potato", "WineBottle", "CellPhone"]
RECEP_OBJECTS = ["CounterTop", "DiningTable", "Cabinet", "Microwave", "CoffeeMachine", "StoveBurner", "Fridge", "GarbageCan", "Chair", "CoffeeTable", "SideTable", "Sofa", "Bed", "Desk", "Sink", "Toilet", ] # "ToiletPaperHanger"]

TRAIN_TASKS = [
    f"pick_and_place_{pick}_{recep}"
    for pick, recep in PICKUPABLE_RECEPTACLE_PAIRS
    if pick not in PICKUP_OBJECTS_FOR_TEST
]
TEST_ONLY_TASKS = [
    f"pick_and_place_{pick}_{recep}"
    for pick, recep in PICKUPABLE_RECEPTACLE_PAIRS
    if pick in PICKUP_OBJECTS_FOR_TEST
]
TOTAL_TASKS = TRAIN_TASKS + TEST_ONLY_TASKS
NUM_TRAIN_TASKS = len(TRAIN_TASKS)
NUM_TEST_TASKS = NUM_VAL_TASKS = len(TOTAL_TASKS)

OBJECT_TYPES_THAT_CAN_HAVE_IDENTICAL_MESHES = [
    "AluminumFoil",
    "CD",
    "Dumbbell",
    "Ladle",
    "Vase",
]

GOOD_4_ROOM_HOUSES = {
    # 1467
    "train": [
        4, 7, 15, 18, 48, 59, 67, 69, 73, 77, 84, 86, 89, 95, 100, 115, 123, 127, 137, 146, 147, 148, 151, 152, 163, 164, 169, 171, 173, 177, 183, 184, 188, 198, 204, 205, 206, 213, 217, 221, 226, 238, 241, 246, 260, 262, 264, 273, 279, 282, 291, 306, 307, 308, 323, 328, 335, 336, 344, 350, 368, 377, 381, 385, 387, 389, 403, 406, 408, 416, 422, 426, 429, 442, 447, 459, 463, 466, 469, 470, 482, 499, 508, 513, 527, 529, 533, 538, 541, 543, 545, 546, 561, 572, 573, 575, 587, 597, 599, 603, 621, 625, 629, 630, 631, 661, 663, 666, 669, 692, 699, 700, 701, 713, 714, 718, 724, 727, 734, 746, 757, 761, 786, 796, 797, 816, 829, 847, 856, 860, 870, 876, 884, 890, 892, 899, 910, 922, 923, 926, 930, 965, 968, 969, 973, 977, 983, 998,
        1002, 1020, 1024, 1031, 1043, 1052, 1057, 1060, 1064, 1075, 1082, 1089, 1091, 1092, 1095, 1096, 1102, 1116, 1118, 1123, 1128, 1129, 1131, 1132, 1133, 1142, 1147, 1161, 1163, 1180, 1182, 1183, 1189, 1204, 1205, 1213, 1220, 1232, 1239, 1245, 1260, 1277, 1283, 1286, 1298, 1299, 1308, 1324, 1332, 1355, 1389, 1405, 1408, 1411, 1412, 1413, 1418, 1428, 1434, 1437, 1442, 1443, 1450, 1452, 1453, 1460, 1475, 1477, 1487, 1489, 1496, 1514, 1519, 1532, 1538, 1540, 1543, 1553, 1556, 1574, 1581, 1584, 1588, 1593, 1594, 1598, 1602, 1604, 1618, 1639, 1644, 1649, 1656, 1665, 1667, 1673, 1687, 1691, 1697, 1708, 1721, 1722, 1745, 1748, 1752, 1759, 1765, 1783, 1788, 1804, 1805, 1806, 1807, 1810, 1816, 1818, 1834, 1836, 1841, 1843, 1844, 1869, 1870, 1872, 1879, 1887, 1897, 1903, 1915, 1916, 1919, 1920, 1926, 1939, 1949, 1960, 1964, 1975, 1991, 1999,
        2003, 2011, 2015, 2040, 2070, 2074, 2087, 2094, 2119, 2140, 2150, 2159, 2162, 2175, 2189, 2202, 2204, 2206, 2214, 2219, 2221, 2222, 2224, 2225, 2228, 2236, 2246, 2247, 2259, 2265, 2272, 2277, 2278, 2284, 2287, 2290, 2309, 2312, 2332, 2345, 2363, 2369, 2423, 2424, 2432, 2438, 2444, 2450, 2464, 2473, 2477, 2479, 2505, 2509, 2524, 2525, 2532, 2547, 2558, 2564, 2565, 2566, 2571, 2572, 2576, 2581, 2603, 2612, 2623, 2628, 2633, 2659, 2664, 2668, 2677, 2693, 2695, 2706, 2718, 2720, 2723, 2737, 2738, 2739, 2740, 2746, 2748, 2757, 2773, 2774, 2776, 2778, 2781, 2803, 2813, 2814, 2822, 2823, 2826, 2839, 2844, 2846, 2850, 2852, 2853, 2865, 2870, 2874, 2875, 2877, 2893, 2920, 2927, 2929, 2947, 2949, 2953, 2965, 2968, 2972, 2976, 2978, 2987, 2991, 2992,
        3000,3005, 3014, 3016, 3018, 3020, 3033, 3035, 3044, 3050, 3053, 3063, 3065, 3066, 3078, 3086, 3093, 3094, 3098, 3110, 3113, 3118, 3126, 3135, 3143, 3150, 3164, 3169, 3173, 3183, 3184, 3189, 3191, 3211, 3214, 3218, 3232, 3241, 3242, 3247, 3252, 3261, 3274, 3282, 3285, 3289, 3291, 3294, 3295, 3302, 3321, 3340, 3345, 3378, 3379, 3386, 3394, 3402, 3411, 3412, 3435, 3438, 3442, 3461, 3463, 3464, 3470, 3473, 3479, 3481, 3482, 3489, 3498, 3506, 3507, 3509, 3518, 3526, 3544, 3545, 3548, 3556, 3570, 3577, 3581, 3585, 3586, 3599, 3608, 3610, 3616, 3621, 3627, 3652, 3655, 3656, 3661, 3667, 3668, 3671, 3672, 3674, 3679, 3696, 3701, 3713, 3716, 3718, 3724, 3729, 3734, 3736, 3742, 3751, 3753, 3772, 3777, 3788, 3793, 3802, 3818, 3820, 3830, 3835, 3841, 3843, 3852, 3872, 3873, 3886, 3894, 3896, 3897, 3899, 3904, 3911, 3912, 3914, 3916, 3944, 3953, 3962, 3963, 3974, 3976, 3981, 3985, 3994,
        4004, 4006, 4010, 4011, 4022, 4038, 4040, 4041, 4050, 4054, 4056, 4075, 4084, 4090, 4093, 4097, 4098, 4100, 4116, 4119, 4120, 4122, 4128, 4133, 4136, 4142, 4145, 4147, 4159, 4161, 4212, 4213, 4216, 4224, 4246, 4249, 4250, 4256, 4264, 4281, 4284, 4290, 4294, 4295, 4304, 4307, 4309, 4310, 4312, 4315, 4330, 4331, 4332, 4333, 4339, 4342, 4349, 4350, 4362, 4363, 4364, 4368, 4370, 4373, 4382, 4387, 4392, 4393, 4409, 4413, 4419, 4422, 4453, 4464, 4467, 4469, 4476, 4479, 4484, 4492, 4495, 4498, 4510, 4533, 4541, 4545, 4551, 4557, 4562, 4563, 4564, 4566, 4584, 4591, 4593, 4598, 4599, 4602, 4608, 4610, 4619, 4622, 4629, 4630, 4636, 4638, 4642, 4648, 4652, 4660, 4671, 4682, 4695, 4706, 4707, 4711, 4723, 4730, 4738, 4743, 4746, 4754, 4760, 4769, 4772, 4790, 4808, 4821, 4828, 4837, 4838, 4841, 4844, 4850, 4856, 4862, 4873, 4876, 4882, 4883, 4885, 4893, 4913, 4915, 4916, 4918, 4925, 4927, 4942, 4958, 4998,
        5007, 5011, 5012, 5015, 5018, 5020, 5021, 5057, 5058, 5065, 5069, 5080, 5091, 5092, 5095, 5105, 5109, 5110, 5115, 5116, 5119, 5121, 5123, 5128, 5130, 5133, 5134, 5141, 5143, 5160, 5162, 5163, 5194, 5195, 5198, 5201, 5204, 5209, 5211, 5222, 5229, 5235, 5239, 5247, 5251, 5253, 5259, 5266, 5273, 5282, 5289, 5319, 5330, 5332, 5336, 5365, 5374, 5381, 5382, 5383, 5391, 5395, 5397, 5400, 5407, 5415, 5425, 5432, 5438, 5442, 5455, 5462, 5464, 5469, 5476, 5478, 5487, 5496, 5509, 5514, 5518, 5522, 5533, 5535, 5539, 5543, 5545, 5551, 5569, 5571, 5587, 5592, 5604, 5623, 5628, 5630, 5643, 5660, 5667, 5671, 5685, 5688, 5689, 5691, 5695, 5700, 5711, 5712, 5715, 5721, 5728, 5730, 5737, 5744, 5746, 5753, 5773, 5776, 5786, 5788, 5792, 5802, 5805, 5815, 5816, 5823, 5825, 5834, 5836, 5846, 5850, 5853, 5861, 5864, 5868, 5869, 5872, 5873, 5874, 5886, 5896, 5900, 5911, 5914, 5931, 5932, 5935, 5940, 5947, 5992, 5995,
        6009, 6020, 6028, 6037, 6039, 6042, 6049, 6050, 6056, 6057, 6058, 6060, 6069, 6086, 6093, 6099, 6100, 6108, 6124, 6127, 6133, 6138, 6143, 6160, 6168, 6174, 6179, 6180, 6183, 6186, 6187, 6190, 6209, 6214, 6216, 6222, 6232, 6238, 6239, 6242, 6243, 6244, 6245, 6248, 6258, 6268, 6269, 6278, 6280, 6285, 6292, 6305, 6314, 6320, 6324, 6325, 6330, 6336, 6345, 6346, 6367, 6380, 6388, 6397, 6402, 6404, 6410, 6438, 6441, 6444, 6447, 6452, 6461, 6479, 6491, 6497, 6500, 6517, 6518, 6521, 6524, 6528, 6540, 6550, 6557, 6562, 6584, 6604, 6607, 6614, 6620, 6630, 6638, 6645, 6659, 6686, 6691, 6717, 6722, 6728, 6730, 6733, 6740, 6754, 6758, 6762, 6770, 6783, 6788, 6795, 6796, 6820, 6822, 6823, 6826, 6832, 6837, 6838, 6841, 6844, 6851, 6857, 6859, 6871, 6872, 6876, 6878, 6880, 6885, 6887, 6890, 6891, 6896, 6913, 6923, 6929, 6939, 6943, 6951, 6952, 6953, 6954, 6972, 6974, 6975, 6976, 6983, 6998,
        7000, 7004, 7008, 7014, 7017, 7023, 7031, 7036, 7037, 7044, 7047, 7054, 7064, 7074, 7075, 7078, 7081, 7085, 7088, 7092, 7095, 7098, 7102, 7103, 7113, 7119, 7123, 7130, 7133, 7137, 7141, 7142, 7144, 7151, 7152, 7158, 7180, 7181, 7191, 7195, 7209, 7213, 7215, 7216, 7233, 7248, 7249, 7251, 7253, 7255, 7256, 7257, 7261, 7268, 7278, 7279, 7292, 7300, 7301, 7307, 7311, 7317, 7329, 7332, 7340, 7349, 7355, 7364, 7385, 7389, 7392, 7395, 7406, 7411, 7419, 7421, 7423, 7424, 7431, 7432, 7435, 7475, 7485, 7488, 7506, 7509, 7510, 7514, 7518, 7522, 7539, 7544, 7546, 7547, 7550, 7551, 7561, 7564, 7567, 7576, 7588, 7593, 7598, 7599, 7604, 7607, 7611, 7616, 7620, 7621, 7627, 7632, 7634, 7643, 7645, 7648, 7650, 7654, 7661, 7662, 7668, 7677, 7681, 7691, 7693, 7719, 7727, 7732, 7751, 7753, 7755, 7756, 7762, 7770, 7772, 7774, 7780, 7790, 7792, 7797, 7802, 7806, 7819, 7828, 7832, 7837, 7839, 7843, 7845, 7860, 7862, 7866, 7878, 7892, 7907, 7938, 7941, 7948, 7951, 7965, 7967, 7968, 7973,
        8005, 8026, 8029, 8034, 8044, 8056, 8064, 8068, 8073, 8077, 8085, 8086, 8090, 8092, 8102, 8109, 8120, 8121, 8136, 8157, 8171, 8173, 8180, 8190, 8196, 8201, 8206, 8208, 8211, 8212, 8221, 8223, 8225, 8230, 8239, 8258, 8261, 8263, 8264, 8269, 8271, 8272, 8280, 8282, 8287, 8293, 8295, 8299, 8310, 8318, 8323, 8325, 8329, 8336, 8341, 8352, 8354, 8372, 8381, 8388, 8391, 8397, 8410, 8412, 8414, 8415, 8416, 8421, 8423, 8429, 8430, 8444, 8458, 8461, 8467, 8483, 8497, 8508, 8509, 8510, 8513, 8521, 8543, 8546, 8547, 8554, 8555, 8558, 8559, 8568, 8581, 8593, 8596, 8600, 8604, 8613, 8614, 8616, 8625, 8636, 8637, 8638, 8639, 8645, 8654, 8662, 8673, 8682, 8685, 8699, 8725, 8741, 8746, 8750, 8763, 8770, 8783, 8791, 8794, 8797, 8823, 8825, 8829, 8831, 8834, 8835, 8838, 8845, 8850, 8855, 8860, 8862, 8866, 8869, 8876, 8878, 8884, 8888, 8894, 8901, 8905, 8915, 8916, 8917, 8921, 8926, 8927, 8932, 8941, 8969, 8974, 8979, 8991, 8997,
        9000, 9015, 9017, 9018, 9022, 9023, 9024, 9025, 9026, 9029, 9034, 9035, 9036, 9037, 9048, 9051, 9058, 9060, 9062, 9065, 9079, 9084, 9094, 9117, 9130, 9149, 9166, 9167, 9189, 9200, 9202, 9204, 9217, 9218, 9227, 9245, 9261, 9263, 9268, 9269, 9289, 9297, 9300, 9302, 9320, 9321, 9344, 9345, 9349, 9361, 9362, 9375, 9383, 9397, 9399, 9402, 9409, 9416, 9419, 9431, 9487, 9491, 9496, 9497, 9506, 9509, 9525, 9529, 9540, 9556, 9560, 9569, 9584, 9595, 9599, 9601, 9618, 9622, 9630, 9631, 9635, 9641, 9643, 9647, 9655, 9659, 9660, 9671, 9677, 9682, 9685, 9700, 9708, 9720, 9738, 9739, 9746, 9761, 9763, 9768, 9778, 9797, 9806, 9807, 9811, 9814, 9817, 9820, 9821, 9830, 9838, 9841, 9842, 9859, 9864, 9865, 9866, 9880, 9884, 9887, 9889, 9896, 9909, 9912, 9918, 9924, 9929, 9938, 9939, 9943, 9947, 9953, 9955, 9965, 9977, 9979, 9982, 9986, 9987,
    ],
    # 143
    "val": [
        11, 13, 18, 27, 39, 43, 57, 63, 65, 67, 77, 86, 89, 101, 102, 110, 114, 116, 125, 136, 137, 144, 155, 159, 171, 179, 184, 185, 195, 197, 200, 201, 211, 221, 234, 243, 265, 266, 271, 296, 310, 311, 312, 324, 327, 335, 338, 344, 349, 350, 372, 378, 379, 384, 390, 397, 399, 406, 411, 413, 422, 427, 430, 434, 436, 443, 446, 469, 471, 474, 476, 487, 488, 501, 521, 526, 528, 546, 547, 575, 579, 581, 587, 603, 607, 609, 613, 627, 637, 639, 640, 644, 654, 659, 667, 679, 690, 691, 693, 699, 700, 706, 729, 732, 733, 736, 737, 749, 751, 757, 761, 763, 772, 777, 779, 798, 804, 831, 832, 837, 843, 849, 862, 864, 865, 879, 886, 889, 893, 897, 899, 906, 915, 922, 931, 935, 938, 939, 940, 958, 961, 969, 974,
    ],
    # 146
    "test": [
        0, 15, 16, 19, 20, 28, 35, 51, 60, 69, 73, 74, 77, 85, 102, 106, 108, 109, 118, 119, 132, 135, 143, 173, 180, 189, 193, 198, 199, 205, 211, 221, 222, 226, 228, 229, 236, 246, 249, 252, 268, 273, 274, 282, 286, 294, 296, 317, 328, 330, 339, 351, 354, 355, 365, 366, 369, 372, 375, 384, 394, 400, 405, 406, 415, 428, 452, 454, 455, 457, 464, 465, 472, 473, 490, 493, 504, 510, 511, 512, 514, 522, 523, 524, 526, 527, 530, 542, 545, 547, 556, 557, 564, 566, 577, 579, 580, 591, 593, 605, 610, 628, 651, 657, 663, 677, 680, 681, 682, 706, 708, 709, 712, 725, 731, 733, 765, 766, 772, 775, 785, 794, 808, 811, 821, 824, 843, 844, 845, 847, 849, 850, 862, 866, 876, 892, 893, 897, 904, 905, 914, 940, 963, 979, 998, 999,
    ],
}

STAGE_TO_MIN_SCENES = {
    "train": 50,
    "val": 10,
    "test": 10,
}

STAGE_TO_DEST_NUM_SCENES = {
    "train": 20,
    "val": 5,
    "test": 5,
}

# METADATA_DIR = os.path.join(STARTER_HOME_SERVICE_DATA_DIR, "metadata")
# all_ready = True
# for stage in ("train", "val", "test"):
#     for cat in ("scenes", "pickupables", "receptacles"):
#         if not os.path.exists(
#             os.path.join(METADATA_DIR, f"{stage}_metadata_{cat}.pkl.gz")
#         ):
#             all_ready = False
#             break
#     if not all_ready:
#         break

# if all_ready:
#     HOME_SERVICE_HOUSES_METADATA = {
#         stage: {
#             cat: compress_pickle.load(
#                 os.path.join(METADATA_DIR, f"{stage}_metadata_{cat}.pkl.gz")
#             )
#             for cat in ("scenes", "pickupables", "receptacles")
#         }
#         for stage in ("train", "val", "test")
#     }

#     STAGE_TO_TASKS = {
#         stage: [
#             f"{stage}_pick_and_place_{pick}_{recep}"
#             for pick, recep in PICKUPABLE_RECEPTACLE_PAIRS
#             if (
#                 pick not in (PICKUP_OBJECTS_FOR_TEST if 'train' in stage else [])
#                 and pick in HOME_SERVICE_HOUSES_METADATA[stage]["pickupables"]
#                 and recep in HOME_SERVICE_HOUSES_METADATA[stage]["receptacles"]
#             )
#         ]
#         for stage in ("train", "val", "test")
#     }

#     STAGE_TO_TASK_TO_SCENES = {
#         stage: {} for stage in STAGE_TO_TASKS
#     }
#     for stage in STAGE_TO_TASKS:
#         for task in STAGE_TO_TASKS[stage]:
#             pick, recep = task.split("_")[-2:]
#             STAGE_TO_TASK_TO_SCENES[stage][task] = [
#                 scene
#                 for scene in HOME_SERVICE_HOUSES_METADATA[stage]["scenes"]
#                 if (
#                     scene in HOME_SERVICE_HOUSES_METADATA[stage]["pickupables"][pick]
#                     and scene in HOME_SERVICE_HOUSES_METADATA[stage]["receptacles"][recep]
#                 )
#             ]

#     STAGE_TO_TASK_TO_NUM_SCENES = {
#         stage: {
#             task: len(STAGE_TO_TASK_TO_SCENES[stage][task])
#             for task in tasks
#         }
#         for stage, tasks in STAGE_TO_TASKS.items()
#     }

    

#     STAGE_TO_VALID_TASKS = {
#         stage: [task for task in tasks if STAGE_TO_TASK_TO_NUM_SCENES[stage][task] >= STAGE_TO_MIN_SCENES[stage]]
#         for stage, tasks in STAGE_TO_TASKS.items()
#     }
#     # train: 363, val: 419, test:442

#     STAGE_TO_VALID_TASK_TO_SCENES = {
#         stage: {
#             task: STAGE_TO_TASK_TO_SCENES[stage][task]
#             for task in STAGE_TO_VALID_TASKS[stage]
#         }
#         for stage in STAGE_TO_TASKS
#     }

#     STAGE_TO_VALID_TASK_TO_NUM_SCENES = {
#         stage: {
#             task: len(STAGE_TO_VALID_TASK_TO_SCENES[stage][task])
#             for task in STAGE_TO_VALID_TASK_TO_SCENES[stage]
#         }
#         for stage in STAGE_TO_VALID_TASK_TO_SCENES
#     }

#     STAGE_TO_VALID_TASK_TO_NUM_SCENES_SORTED = {
#         stage: dict(sorted(STAGE_TO_VALID_TASK_TO_NUM_SCENES[stage].items(), key=lambda item: item[1]))
#         for stage in STAGE_TO_VALID_TASK_TO_SCENES
#     }

#     STAGE_TO_SCENE_TO_TASKS = {
#         stage: {} for stage in STAGE_TO_TASKS
#     }
#     for stage in STAGE_TO_TASKS:
#         for task, scenes in STAGE_TO_TASK_TO_SCENES[stage].items():
#             for scene in scenes:
#                 if scene not in STAGE_TO_SCENE_TO_TASKS[stage]:
#                     STAGE_TO_SCENE_TO_TASKS[stage][scene] = []
#                 STAGE_TO_SCENE_TO_TASKS[stage][scene].append(task)
#     STAGE_TO_SCENE_TO_NUM_TASKS = {
#         stage: {
#             scene: len(tasks)
#             for scene, tasks in STAGE_TO_SCENE_TO_TASKS[stage].items()
#         }
#         for stage in STAGE_TO_TASKS
#     }

#     STAGE_TO_SCENE_TO_VALID_TASKS = {
#         stage: {
#             scene: [
#                 task
#                 for task in STAGE_TO_SCENE_TO_TASKS[stage][scene]
#                 if task in STAGE_TO_VALID_TASKS[stage]
#             ]
#             for scene in STAGE_TO_SCENE_TO_TASKS[stage]
#         }
#         for stage in STAGE_TO_SCENE_TO_TASKS
#     }
#     STAGE_TO_SCENE_TO_NUM_VALID_TASKS = {
#         stage: {
#             scene: len(tasks)
#             for scene, tasks in STAGE_TO_SCENE_TO_VALID_TASKS[stage].items()
#         }
#         for stage in STAGE_TO_TASKS
#     }

#     STAGE_TO_SCENE_TO_NUM_VALID_TASKS_SORTED = {
#         stage: dict(sorted(STAGE_TO_SCENE_TO_NUM_VALID_TASKS[stage].items(), key=lambda item: item[1], reverse=True))
#         for stage in STAGE_TO_SCENE_TO_NUM_VALID_TASKS
#     }

#     STAGE_TO_MIN_SCENES_COVERING_ALL_VALID_TASKS = {}
#     for stage in STAGE_TO_VALID_TASK_TO_NUM_SCENES_SORTED:
#         scene_set = set()
#         task_set = set()
#         all_tasks = False
#         for task in STAGE_TO_VALID_TASK_TO_NUM_SCENES_SORTED[stage]:
#             for scene in STAGE_TO_VALID_TASK_TO_SCENES[stage][task]:
#                 if scene not in scene_set:
#                     scene_set.add(scene)
                
#                     for t in STAGE_TO_SCENE_TO_VALID_TASKS[stage][scene]:
#                         if t not in task_set:
#                             task_set.add(t)
#                         if len(task_set) == len(STAGE_TO_VALID_TASKS[stage]):
#                             all_tasks = True
#                             break
#                 if all_tasks:
#                     break
#             if all_tasks:
#                 break
#         STAGE_TO_MIN_SCENES_COVERING_ALL_VALID_TASKS[stage] = scene_set

#     STAGE_TO_NUM_MIN_SCENES_COVERING_ALL_VALID_TASKS = {
#         stage: len(STAGE_TO_MIN_SCENES_COVERING_ALL_VALID_TASKS[stage])
#         for stage in STAGE_TO_TASKS
#     }
#     # train: 20, val: 19, test:18

#     STAGE_TO_DEST_NUM_SCENES = {
#         "train": 100,
#         "val": 10,
#         "test": 10,
#     }

#     # STAGE_TO_SCENES_COVERING_ALL_VALID_TASKS
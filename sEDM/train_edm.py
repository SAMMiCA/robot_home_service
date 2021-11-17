import re
import pickle
from sedm import EDM, sEDM
from data_fasttext import new_ft_dict

train_file = open("train_unique_data.pkl", "rb")
data_dict_train = pickle.load(train_file)

# model = EDM()
model = sEDM()

for i, (k, v) in enumerate(data_dict_train.items()):
    goal_wv = []
    goal_wv_entire = []
    object_wv = []
    object_wv_entire = []
    instr_wv = []
    instr_wv_entire = []
    print(i)
    # if i == 6:
    #     break
    goals = v['goal']
    objects = v['objects']
    instr = v['instruction']
    for g in goals:
        go = g.split(" ")
        for w in go:
            w = re.sub('[\W_]+', '', w.lower())
            wvec = new_ft_dict[w]
            goal_wv.append(list(wvec))
        goal_wv_entire.append(goal_wv)
        goal_wv = []
    for o in objects:
        obj = o.split(" ")
        for w in obj:
            w = re.sub('[\W_]+', '', w.lower())
            wvec = new_ft_dict[w]
            object_wv.append(list(wvec))
        object_wv_entire.append(object_wv)
        object_wv = []
    for i in instr:
        ins = i.split(" ")
        for w in ins:
            w = re.sub('[\W_]+', '', w.lower())
            wvec = new_ft_dict[w]
            instr_wv.append(list(wvec))
        instr_wv_entire.append(instr_wv)
        instr_wv = []
    model.train(goal_wv_entire, object_wv_entire, instr_wv_entire)

print("done training")
model.save(filename="weight_unique.pickle")





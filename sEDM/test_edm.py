import re
from sedm import EDM, sEDM
from data_fasttext import new_ft_dict
from data_bert import match_most_similar


def replace_action(seq):
    seq = seq.replace('navigate', 'Navigate') if 'navigate' in seq else seq
    seq = seq.replace('pick up', 'Pickup') if 'pick up' in seq else seq
    seq = seq.replace('put', 'Put') if 'put' in seq else seq
    return seq

class sEDM_model(object):
    def __init__(self):
        self.sedm = sEDM()
        self.sedm.load(filename="weight_unique.pickle")
        # self.pair_cos = match_most_similar(metric="cosine")

    def inference(self, target_object, target_place):
        task_plan = []
        pair = match_most_similar(metric="cosine", target_object=target_object, target_place=target_place)
        goal_wv = []
        goal_wv_entire = []
        object_wv = []
        object_wv_entire = []
        test_order = pair[1]
        train_order = pair[2]
        test_obj = pair[3]
        train_obj = pair[4]
        test_obj = [ob.replace('openable ', '') if 'openable' in ob else ob for ob in test_obj]
        train_obj = [ob.replace('openable ', '') if 'openable' in ob else ob for ob in train_obj]
        print("Test order: {}\n".format(test_order))
        # print("Train order: {}".format(train_order))
        if len(test_obj) == len(train_obj):
            diff_test_tr = [(i, j) for i, j in zip(test_obj, train_obj) if i != j]
        else:
            print("***sequence cannot be retrieved***")
            print("\n\n")
        for w in train_order.split(" "):
            w = re.sub('[\W_]+', '', w.lower())
            wvec = new_ft_dict[w]
            goal_wv.append(list(wvec))
        goal_wv_entire.append(goal_wv)
        goal_wv = []
        for o in train_obj:
            for w in o.split(" "):
                w = re.sub('[\W_]+', '', w.lower())
                wvec = new_ft_dict[w]
                object_wv.append(list(wvec))
            object_wv_entire.append(object_wv)
            object_wv = []
        sequence = self.sedm.test([goal_wv_entire, object_wv_entire])
        # sequence = model.test(goal_wv_entire)
        # print(sequence)
        for seq in sequence:
            seq = replace_action(seq)
            for kw in diff_test_tr:
                if kw[1] in seq:
                    seq = seq.replace(kw[1], kw[0])
            action = seq.split(" ")[0]
            target = seq.split(" ")[-2]
            triple = (action, target, None)
            if action == "Put":
                other_target = seq.split(" ")[1]
                triple = (action, other_target, target)
            task_plan.append(triple)
        return task_plan


if __name__ == '__main__':
    model = sEDM_model()
    model.inference(target_object="Spatula", target_place="SinkBasin")




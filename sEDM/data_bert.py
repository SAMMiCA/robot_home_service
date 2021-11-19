import pickle
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

train_file = open("train_unique_data.pkl", "rb")
test_file = open("test_unique_data.pkl", "rb")
data_dict_train = pickle.load(train_file)
data_dict_test = pickle.load(test_file)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states=True,  # Whether the model returns all hidden-states.
                                  )

two_word_dict = {'apple sliced': 'AppleSliced',
                 'arm chair': 'ArmChair',
                 'bathtub basin': 'BathtubBasin',
                 'wine bottle': 'WineBottle',
                 'spray bottle': 'SprayBottle',
                 'soap bottle': 'SoapBottle',
                 'bread sliced': 'BreadSliced',
                 'butter knife': 'ButterKnife',
                 'desk lamp': 'DeskLamp',
                 'dog bed': 'DogBed',
                 'egg cracked': 'EggCracked',
                 'floor lamp': 'FloorLamp',
                 'towel holder': 'TowelHolder',
                 'paper towel roll': 'PaperTowelRoll',
                 'hand towel': 'HandTowel',
                 'hand towel holder': 'HandTowelHolder',
                 'lettuce sliced': 'LettuceSliced',
                 'potato sliced': 'PotatoSliced',
                 'sink basin': 'SinkBasin',
                 'tissue box': 'TissueBox',
                 'tomato sliced': 'TomatoSliced'}

def check_two_word(target_object, target_place):
    if target_object in two_word_dict.values():
        target_object = [k for k, v in two_word_dict.items() if v == target_object][0]
    if target_place in two_word_dict.values():
        target_place = [k for k, v in two_word_dict.items() if v == target_place][0]
    return target_object, target_place


def dataload_train(v):
    input_ids_objects = []
    goals = v["goal"][0].lower()
    objects = v["objects"][:]

    encoded_goals = tokenizer.encode_plus(text=goals,  # the sentence to be encoded
                                    add_special_tokens=True,  # Add [CLS] and [SEP]
                                    max_length=8,  # maximum length of a sentence
                                    pad_to_max_length=True,  # Add [PAD]s
                                    return_attention_mask=True,  # Generate the attention mask
                                    return_tensors='pt')  # ask the function to return PyTorch tensors)
    input_ids_goals = encoded_goals['input_ids']
    attn_mask_goals = encoded_goals['attention_mask']

    for obj in objects:
        encoded_objects = tokenizer.encode_plus(text=obj.lower(),  # the sentence to be encoded
                                          add_special_tokens=True,  # Add [CLS] and [SEP]
                                          max_length=4,  # maximum length of a sentence
                                          pad_to_max_length=True,  # Add [PAD]s
                                          return_attention_mask=True,  # Generate the attention mask
                                          return_tensors='pt')  # ask the function to return PyTorch tensors)
        input_ids_objects.append(encoded_objects['input_ids'])
    input_ids_objects = torch.cat(input_ids_objects, dim=0)
    attn_mask_objects = encoded_objects['attention_mask']

    model.eval()
    with torch.no_grad():
        goals_emb = model(input_ids_goals, attn_mask_goals)[0]
        objects_emb = model(input_ids_objects, attn_mask_objects)[0]
    return (goals, goals_emb), (objects, objects_emb)

def dataload_test(target_object, target_place):
    input_ids_objects = []
    goal = "pick {} place {}".format(target_object, target_place)
    objects_orig = [target_object, target_place]
    target_object, target_place = check_two_word(target_object, target_place)
    objects = [target_object, target_place]
    encoded_goal = tokenizer.encode_plus(text=goal.lower(),  # the sentence to be encoded
                                          add_special_tokens=True,  # Add [CLS] and [SEP]
                                          max_length=8,  # maximum length of a sentence
                                          pad_to_max_length=True,  # Add [PAD]s
                                          return_attention_mask=True,  # Generate the attention mask
                                          return_tensors='pt')  # ask the function to return PyTorch tensors)
    input_ids_goal = encoded_goal['input_ids']
    attn_mask_goal = encoded_goal['attention_mask']

    for obj in objects:
        encoded_objects = tokenizer.encode_plus(text=obj.lower(),  # the sentence to be encoded
                                          add_special_tokens=True,  # Add [CLS] and [SEP]
                                          max_length=4,  # maximum length of a sentence
                                          pad_to_max_length=True,  # Add [PAD]s
                                          return_attention_mask=True,  # Generate the attention mask
                                          return_tensors='pt')  # ask the function to return PyTorch tensors)
        input_ids_objects.append(encoded_objects['input_ids'])
    input_ids_objects = torch.cat(input_ids_objects, dim=0)
    attn_mask_objects = encoded_objects['attention_mask']

    model.eval()
    with torch.no_grad():
        goal_emb = model(input_ids_goal, attn_mask_goal)[0]
        objects_emb = model(input_ids_objects, attn_mask_objects)[0]
    return (goal, goal_emb), (objects_orig, objects_emb)


def build_train_dict():
    """
    take the mean of the sequence to reduce the dimension to (n_sample, feature_dim) for cosine sim & euclidean
    """
    emb_dict_train = {}
    for i, v_tr in enumerate(data_dict_train.values()):
        emb_dict_train[str(i)] = {}
        (goals, goals_emb), (objects, objects_emb) = dataload_train(v_tr)   # menu, ingre, recipe has different sequence length -> to reduce dimension, they need to be equalized
        emb_dict_train[str(i)]['order'] = goals
        emb_dict_train[str(i)]['objects'] = objects
        goals_emb_mean_tr = torch.mean(goals_emb, 1)       # take an average across the sequence length: (2,16,768) -> (2, 768)
        objects_emb_mean_tr = torch.mean(objects_emb, 1)
        goal_obj_concat = torch.mean(torch.cat((goals_emb_mean_tr, objects_emb_mean_tr), 0), 0)
        emb_dict_train[str(i)]['emb'] = goal_obj_concat
    return emb_dict_train


def build_test_dict(target_object, target_place):
    """
    take the mean of the sequence to reduce the dimension to (n_sample, feature_dim) for cosine sim & euclidean
    """
    emb_dict_test = {}
    (goal, goal_emb), (objects, objects_emb) = dataload_test(target_object, target_place)
    emb_dict_test['order'] = goal
    emb_dict_test['objects'] = objects
    goal_emb_mean = torch.mean(goal_emb, 1)
    objects_emb_mean = torch.mean(objects_emb, 1)
    goal_obj_concat = torch.mean(torch.cat((goal_emb_mean, objects_emb_mean), 0), 0)
    emb_dict_test['emb'] = goal_obj_concat
    return emb_dict_test


def match_most_similar(metric, target_object, target_place):
    """
    match unseen order in test data to the most similar order in train data
    and return the pair of (similarity score, test order, train order)
    """
    emb_dict_train = build_train_dict()
    emb_dict_test = build_test_dict(target_object, target_place)
    test_order = emb_dict_test['order']
    test_obj = emb_dict_test['objects']
    test_emb = emb_dict_test['emb']
    all_cos = []
    all_euc = []
    for j, (k_tr, v_tr) in enumerate(emb_dict_train.items()):
        train_order = emb_dict_train[k_tr]['order']
        train_obj = emb_dict_train[k_tr]['objects']
        train_emb = emb_dict_train[k_tr]['emb']
        if metric == "cosine":
            cos = cosine_similarity(test_emb.unsqueeze(0), train_emb.unsqueeze(0))
            all_cos.append((cos, test_order, train_order, test_obj, train_obj))
        elif metric == "euclidean":
            euc = euclidean_distances(test_emb.unsqueeze(0), train_emb.unsqueeze(0))
            all_euc.append((euc, test_order, train_order, test_obj, train_obj))
    if metric == "cosine":
        most_similar = max(all_cos)
    elif metric == "euclidean":
        most_similar = min(all_euc)
    return most_similar


def find_mismatch():
    kw_pair = []
    pair_cos = match_most_similar(metric="cosine")
    for pair in pair_cos:
        goal_test, goal_train = pair[1].split(" "), pair[2].split(" ")
        test_kw = [x for x in goal_test if x not in goal_train]
        train_kw = [x for x in goal_train if x not in goal_test]
        kw_pair.append((test_kw[0], train_kw[0]))
    return kw_pair


def replace_kw(test_kw, train_kw, train_instr):     # test_kw & train_kw each key word **not list**
    sequence = []
    output = []
    for seq in train_instr:
        for k, v in two_word_dict.items():
            if k in seq:
                seq = seq.replace(k, v)
        sequence.append(seq)
    replace = ''
    replace2 = ''
    for tr_kw in train_kw:
        replace = replace + tr_kw + ' '
    for te_kw in test_kw:
        replace2 = replace2 + te_kw + ' '
    for s in sequence:
        if replace in s:
            s = s.replace(replace, replace2)
        output.append(s)
    return output


if __name__ == '__main__':
    goal = "pick spatula place sink"
    # test_dict = build_test_dict(goal)
    pair_cos = match_most_similar(metric="cosine", target_object="Spatula", target_place="SinkBasin")
    pair_euc = match_most_similar(metric="euclidean", target_object="Spatula", target_place="SinkBasin")
    print(pair_cos)
    print(pair_euc)

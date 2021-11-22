import torch
from torchnlp.word_to_vector import FastText

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import re
import pickle
import numpy as np
import os
from pathlib import Path

train_file = open(os.path.join(os.path.abspath(os.path.dirname(Path(__file__))), "train_unique_data.pkl"), "rb")
test_file = open(os.path.join(os.path.abspath(os.path.dirname(Path(__file__))), "test_unique_data.pkl"), "rb")
data_dict_train = pickle.load(train_file)
data_dict_test = pickle.load(test_file)

vectors = FastText()


def build_ft_dict():
    ft_dict = {}
    for v_tr in data_dict_train.values():
        goals = v_tr["goal"][0]
        objects = v_tr["objects"][:]
        instructions = v_tr["instruction"][:]
        # print(goals)
        # print(objects)
        # print(instructions)
        for g_w in goals.split(" "):
            g_w = g_w.lower()
            g_w = re.sub('[\W_]+', '', g_w)
            if not g_w in ft_dict.keys():
                ft = vectors[g_w]
                ft_dict[g_w] = ft
        for obj in objects:
            for o_w in obj.split(" "):
                o_w = o_w.lower()
                o_w = re.sub('[\W_]+', '', o_w)
                if not o_w in ft_dict.keys():
                    ft = vectors[o_w]
                    ft_dict[o_w] = ft
        for ins in instructions:
            for i_w in ins.split(" "):
                i_w = i_w.lower()
                i_w = re.sub('[\W_]+', '', i_w)
                if not i_w in ft_dict.keys():
                    ft = vectors[i_w]
                    ft_dict[i_w] = ft

    for v_te in data_dict_test.values():
        goals_te = v_te["goal"][0]
        for g_w in goals_te.split(" "):
            g_w = g_w.lower()
            g_w = re.sub('[\W_]+', '', g_w)
            if not g_w in ft_dict.keys():
                ft = vectors[g_w]
                ft_dict[g_w] = ft

    return ft_dict


def dim_reduction(features_variable):
    standarized_data = StandardScaler().fit_transform(features_variable)    # features_variable: (n_samples, n_features)
    model = TSNE(n_components=4, perplexity=50, n_iter=5000, random_state=0, method="exact")
    tsne_data = model.fit_transform(standarized_data)
    return tsne_data


def build_new_dict(ft_dict):
    new_ft_dict = {}
    embs = []
    for v in ft_dict.values():
        embs.append(v.detach().cpu().numpy())
    ft_dim_red = dim_reduction(np.array(embs))

    for i, k in enumerate(ft_dict.keys()):
        new_ft_dict[k] = ft_dim_red[i]
    return new_ft_dict

ft_dict = build_ft_dict()
new_ft_dict = build_new_dict(ft_dict)

if __name__ == '__main__':
    ft_dict = build_ft_dict()
    # print(ft_dict)
    new_ft_dict = build_new_dict(ft_dict)
    print(new_ft_dict)

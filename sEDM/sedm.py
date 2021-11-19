from em_drn import EM_DRN, EM_DRN_recipe, EM_sDRN, EM_sDRN_recipe
from edmap import EDMAP
import numpy as np
import pickle

class EDM(object):
    def __init__(self):
        self.em_drn1 = EM_DRN()             # menu
        self.em_drn2 = EM_DRN()            # ingredient
        self.em_drn3 = EM_DRN_recipe()    #recipe
        self.edmap = EDMAP()

    def train(self, X1, X2, X3):
        Y1 = self.em_drn1.train(X1)
        Y2 = self.em_drn2.train(X2)
        Y3 = self.em_drn3.train(X3)
        Xn = []
        Xn.append(np.sum(np.array([Y2])[0], axis=0))
        Xn.append(Y3[0])
        Xn = np.array(Xn)
        label = Y1[0]
        self.edmap.train(Xn, label)

    def test(self, cue):
        if cue.shape[0] == 1:
            Y2 = self.em_drn2.test(cue[0])
            Xn = np.array([np.sum(Y2, axis=0)])
            label = []
        else:
            Y1 = self.em_drn1.test(cue[0])
            Y2 = self.em_drn2.test(cue[1])
            Xn = np.sum(Y2[0])
            label = Y1[0]

        pred_Y1, pred_Y3 = self.edmap.test(Xn, label)

        for i in range(pred_Y1.shape[0]):
            self.em_drn1.readout(pred_Y1[i], "food_type")
            self.em_drn3.readout(pred_Y3[i], "recipe")


class sEDM(object):
    def __init__(self):
        self.em_sdrn1 = EM_sDRN()             # menu
        self.em_sdrn2 = EM_sDRN()            # ingredient
        self.em_sdrn3 = EM_sDRN_recipe()    #recipe
        self.edmap = EDMAP()

    def train(self, X1, X2, X3):
        Y1 = self.em_sdrn1.train(X1)
        Y2 = self.em_sdrn2.train(X2)
        Y3 = self.em_sdrn3.train(X3)
        Xn = []
        Xn.append(np.sum(np.array([Y2])[0], axis=0))
        Xn.append(Y3[0])
        Xn = np.array(Xn)
        label = Y1[0]
        self.edmap.train(Xn, label)

    def test(self, cue):
        if type(cue) == list:
            cue = np.array(cue)
        if cue.shape[0] == 1:
            ##### if input in only menu (order)
            Y1 = self.em_sdrn1.test(cue)
            if Y1.shape[0] == 1:
                Xn = np.array([np.sum(Y1, axis=1)])
            else:
                Xn = np.array([np.sum(Y1, axis=0)])
            
            ##### if input is only ingredients (objects)
            # Y2 = self.em_sdrn2.test(cue)
            # if Y2.shape[0] == 1:
            #     Xn = np.array([np.sum(Y2, axis=1)])
            # else:
            #     Xn = np.array([np.sum(Y2, axis=0)])
            label = []
        else:
            Y1 = self.em_sdrn1.test(cue[0])
            Y2 = self.em_sdrn2.test(cue[1])
            Xn = np.array([np.sum([Y2][0], axis=0)])
            label = Y1

        pred_Y1, pred_Y3 = self.edmap.test(Xn, label)
        count=0
        for i in range(pred_Y1.shape[0]):
            count+=1
            self.em_sdrn1.readout(pred_Y1[i], "food_type")
            recipe = self.em_sdrn3.readout(pred_Y3[i], "recipe")
        return recipe

    def save(self, filename):
        data = {
            'em_sdrn1': self.em_sdrn1, # menu
            'em_sdrn2': self.em_sdrn2, # ingredient
            'em_sdrn3': self.em_sdrn3, # recipe
            'edmap': self.edmap
        }

        with open(filename, 'wb') as f:
            pickle.dump(data, f, pickle.DEFAULT_PROTOCOL)

    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.em_sdrn1 = data['em_sdrn1']  # menu
        self.em_sdrn2 = data['em_sdrn2']  # ingredient
        self.em_sdrn3 = data['em_sdrn3']  # recipe
        self.edmap = data['edmap']
        del data

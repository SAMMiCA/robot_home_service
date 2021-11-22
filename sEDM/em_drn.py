import numpy as np
from sEDM.deepART import deepART
from sEDM.DRN import DRN
from sEDM.SDRN import sDRN
from sEDM.data_fasttext import new_ft_dict


class EM_DRN(object):
    def __init__(self):
        self.numLayer = 1
        # self.sdrn = sDRN()
        self.drn = DRN()
        self.deepArt = deepART()

    def train(self, X):
        Y = self.drn.train(X)
        Y = self.deepArt.train(Y)
        return Y

    def test(self, X):
        Y = self.drn.testDRN(X)
        Y = self.deepArt.testDeepART(Y)
        return Y

    def readout(self, Y, name):
        YY = self.deepArt.readout(Y)
        X = self.drn.readout(YY)
        for i in range(X.shape[0]):
            naturalLang = ""
            for j in range(X[i].shape[0]):
                for k in range(len(new_ft_dict)):
                    if np.sum(abs(X[i][j] - np.array(new_ft_dict[list(new_ft_dict)[k]]))) < 1e-6:
                        naturalLang = naturalLang + list(new_ft_dict)[k] + " "
            print(name, ": ", naturalLang)
        print(" ")


class EM_DRN_recipe(object):
    def __init__(self):
        self.numLayer = 2
        self.drn = DRN()
        self.deepArt1 = deepART()
        self.deepArt2 = deepART()

    def train(self, X):
        Y = self.drn.train(X)
        Y = self.deepArt1.train(Y)
        Y = self.deepArt2.train([Y])
        return Y

    def test(self, X):
        Y = self.drn.testDRN(X)
        Y = self.deepArt1.testDeepART(Y)
        Y = self.deepArt2.testDeepART(Y)
        return Y

    def readout(self, Y, name):
        YY = self.deepArt2.readout(Y)
        YY = self.deepArt1.readout(YY)
        X = self.drn.readout(YY)
        for i in range(X.shape[0]):
            naturalLang = ""
            for j in range(X[i].shape[0]):
                for k in range(len(new_ft_dict)):
                    if np.sum(abs(X[i][j] - np.array(new_ft_dict[list(new_ft_dict)[k]]))) < 1e-6:
                        naturalLang = naturalLang + list(new_ft_dict)[k] + " "
            print(name, ": ", naturalLang)
        print(" ")

class EM_sDRN(object):
    def __init__(self):
        self.numLayer = 1
        self.sdrn = sDRN()
        self.deepArt = deepART()

    def train(self, X):
        Y = self.sdrn.train(X)
        Y = self.deepArt.train(Y)
        return Y

    def test(self, X):
        Y = self.sdrn.testDRN(X)
        Y = self.deepArt.testDeepART(Y)
        return Y

    def readout(self, Y, name):
        YY = self.deepArt.readout(Y)
        X = self.sdrn.readout(YY)
        for i in range(X.shape[0]):
            naturalLang = ""
            for j in range(X[i].shape[0]):
                for k in range(len(new_ft_dict)):
                    if np.sum(abs(X[i][j] - np.array(new_ft_dict[list(new_ft_dict)[k]]))) < 1e-6:
                        naturalLang = naturalLang + list(new_ft_dict)[k] + " "
            # print(name, ": ", naturalLang)
        # print(" ")


class EM_sDRN_recipe(object):
    def __init__(self):
        self.numLayer = 2
        self.sdrn = sDRN()
        self.deepArt1 = deepART()
        self.deepArt2 = deepART()

    def train(self, X):
        Y = self.sdrn.train(X)
        Y = self.deepArt1.train(Y)
        Y = self.deepArt2.train([Y])
        return Y

    def test(self, X):
        Y = self.sdrn.testDRN(X)
        Y = self.deepArt1.testDeepART(Y)
        Y = self.deepArt2.testDeepART(Y)
        return Y

    def readout(self, Y, name):
        recipe = []
        YY = self.deepArt2.readout(Y)
        YY = self.deepArt1.readout(YY)
        X = self.sdrn.readout(YY)
        for i in range(X.shape[0]):
            naturalLang = ""
            for j in range(X[i].shape[0]):
                for k in range(len(new_ft_dict)):
                    if np.sum(abs(X[i][j] - np.array(new_ft_dict[list(new_ft_dict)[k]]))) < 1e-6:
                        naturalLang = naturalLang + list(new_ft_dict)[k] + " "
            # print(name, ": ", naturalLang)
            recipe.append(naturalLang)
        # print(" ")
        return recipe


if __name__ == '__main__':
    x = [[[1.0, 0, 0, 0], [1.0, 0, 0, 0], [0, 1.0, 0, 0]], [[0, 0, 1.0, 0], [0, 0, 0, 1.0]]]
    x = np.array(x)
    model = deepART()
    model.train(x)
    Y = model.testDeepART(x)
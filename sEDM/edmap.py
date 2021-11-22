import numpy as np
from sEDM.utils import fuzzy_or, fuzzy_and


class EDMAP(object):
    def __init__(self, lr=1, rho=0.1, channel=2, alpha=1e-6, numCol=0):
        self.lr = lr  # learning rate
        self.rho = rho  # rho parameter
        self.channel = channel
        self.gamma = [0.5, 0.5]
        self.alpha = alpha  # parameter for node activation
        self.numCol = numCol
        self.numCell = []
        self.X = []
        self.Y = []
        self.cY = []
        self.w = []
        self.label = []

    def learningMAP(self, x, label):
        normX = []
        for i in range(x.shape[0]):
            normX.append(np.append(np.array(x[i]), 1-np.array(x[i])))
        normX = np.array(normX)
        self.X = normX
        self.label = np.round(label)
        label_ind = np.argmax(self.label)
        if len(self.w) < self.label.size:       # self.w.shape[0] < self.label.shape[0]
            if len(self.w) == 0:
                self.w = [[] for _ in range(label_ind+1)]
                self.w[len(self.label) - 1] = np.array([self.X])  # new column is added
                self.w = np.array(self.w)
            elif len(self.w) < label_ind + 1:
                self.w = list(self.w)
                for _ in range(label_ind+1-len(self.w)):
                    self.w.append([])
                wl = np.empty(shape=(1, self.X.shape[0]), dtype=object)
                for i in range(wl.shape[1]):
                    wl[0][i] = self.X[i]
                self.w[label_ind] = wl
                self.w = np.asarray(self.w)
            else:
                wl_new = list(self.w)
                wl = np.empty(shape=(1,self.X.shape[0]), dtype=object)
                for i in range(wl.shape[1]):
                    wl[0][i] = self.X[i]
                wl_new.append(np.asarray(list(wl)))
                self.w = np.asarray(wl_new)

            self.numCol += 1
            self.numCell.append(1)
        else:
            self.activateCell(label_ind)
            cell_index = np.array(self.cY).argsort()[::-1]
            for i in range(self.numCell[label_ind]):
                ind = cell_index[i]
                resonance = 0
                for j in range(self.channel):
                    m = np.sum(fuzzy_and(self.w[label_ind][ind][j], self.X[j]) / np.sum(self.X[j]))
                    if m >= self.rho:
                        resonance += 1
                    else:
                        break
                if resonance == self.channel:
                    for j in range(self.channel):
                        self.w[label_ind][ind][j] = (1-self.lr) * self.w[label_ind][ind][j] + self.lr * fuzzy_and(self.X[j], self.w[label_ind][ind][j])
                    break
                elif i+1 == self.numCell[label_ind]:
                    wl = list(self.w)
                    wl_new = list(self.w[label_ind])
                    if len(self.X.shape) == 2:
                        wl_new.append(list(self.X))
                        wl[label_ind] = wl_new
                        self.w = np.array(wl)
                    elif len(self.X.shape) == 1:
                        wl_new.append(np.asarray(list(self.X)))
                        wl[label_ind] = np.asarray(wl_new)
                        self.w = np.array(wl)
                    self.numCell[label_ind] += 1
        self.Y = self.label

    def activateCell(self, label):
        self.cY = []
        act_k = [[] for _ in range(self.channel)]
        for i in range(self.w[label].shape[0]):
            for k in range(self.channel):
                act_k[k] = self.gamma[k] * (np.sum(fuzzy_and(self.X[k], self.w[label][i][k])) / (self.alpha + np.sum(self.w[label][i][k])))
            self.cY.append(np.sum(act_k))

    def zeroPadding(self, point):
        if len(self.w) != 0:
            for i in range(self.channel):
                if self.X[i].shape[0] < point[i].shape[0] * 2:
                    numZeros = int(point[i].shape[0] - self.X[i].shape[0] / 2)
                    for j in range(self.numCol):
                        for k in range(self.numCell[j]):
                            if len(self.w[j][k][i].shape) == 1:
                                half = self.w[j][k][i].shape[0]//2
                                new_w = np.hstack((np.array([self.w[j][k][i][:half]]), \
                                                   np.zeros((np.array([self.w[j][k][i]]).shape[0], numZeros)),
                                                   np.array([self.w[j][k][i][half:]]),
                                                   np.ones((np.array([self.w[j][k][i]]).shape[0], numZeros))))
                            new_w = np.squeeze(new_w)
                            self.w[j][k][i] = new_w



    def train(self, X, label):
        self.zeroPadding(X)
        self.learningMAP(X, label)

    def test(self, X, label):
        start = 0
        pred_Y1 = []
        pred_Y3 = []
        if len(label) == 0:
            fin = self.numCol
        else:
            ind = np.argmax(label)
            start = ind
            fin = ind + 1
        # T_tmp = np.zeros((self.numCol, 3))  # 각 order 마다 3개의 instruction 존재 (order 개수 = self.numCol (= 7))
        for i in range(start, fin):
            for j in range(self.numCell[i]):
                half1 = self.w[i][j][0].shape[0]//2
                if np.sum(self.w[i][j][0][:half1]) != 0:
                    T = np.sum(fuzzy_and(X[0], self.w[i][j][0][:half1])) / np.sum(self.w[i][j][0][:half1])
                else:
                    T = (np.sum(fuzzy_and(X[0], self.w[i][j][0][:half1])) + 1) / (np.sum(self.w[i][j][0][:half1]) + 1)
                # T_tmp[i][j] = T
                if T >= self.rho:
                    pY1 = np.zeros((1, self.label.size))
                    pY1[0][i] = 1
                    half2 = self.w[i][j][1].shape[0] // 2
                    if len(pred_Y1) == 0:
                        pred_Y1 = np.array([pY1])
                        pred_Y3 = np.array([[self.w[i][j][1][:half2]]])
                    else:
                        pred_Y1 = np.array(np.vstack((pred_Y1, [pY1])))
                        pred_Y3 = np.array(np.vstack((pred_Y3, [[self.w[i][j][1][:half2]]])))
        # print(T_tmp)
        return np.array(pred_Y1), np.array(pred_Y3)


if __name__ == '__main__':
    x = np.array([[1.0, 1.0], [1.0]])
    label = np.array([1.0])
    model = EDMAP()
    model.train(x, label)



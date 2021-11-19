import numpy as np
from utils import fuzzy_and, fuzzy_or


class deepART(object):
    def __init__(self, lr=1, rho=1, category=0, channel=1, gamma=1):
        self.lr = lr  # learning rate
        self.rho = rho  # rho parameter
        self.category = category
        self.channel = channel
        self.gamma = gamma
        self.X = []
        self.Y = []
        self.w = None
        self.maxPosNum = []
        self.numEvent = []

    def seq_encoding(self, X):
        # set weights
        iw = 1
        bw = 2
        ow = 1
        ix = np.zeros((1, X.shape[1]))
        bx = np.zeros((1, X.shape[1]))
        oy = np.zeros((1, X.shape[1]))
        # sequence encoding
        for i in range(X.shape[0]):
            if i != 0:
                bx = ow * oy
            ix = X[i]
            oy = iw * ix + bw * bx
        maxY = np.max(oy)
        maxPosNum = len(str(int(maxY)))
        norm_oy = oy / 10 ** (maxPosNum)
        sample = norm_oy
        numEpisode = X.shape[0]
        return sample, maxPosNum, numEpisode


    def zero_padding(self, point):
        if self.w is not None:
            if self.X.shape[0] < point.shape[1] * 2:
                numZeros = int(point.shape[1] - self.X.shape[0] / 2)
                self.X = np.zeros((point.shape[1]*2))
                new_w = np.hstack((self.w[:,:int(self.w.shape[1]/2)], np.zeros((self.w.shape[0], numZeros)), self.w[:,int(self.w.shape[1]/2):], np.ones((self.w.shape[0], numZeros))))
                self.w = new_w

    def learningART(self, sample, maxPosNum, numEpisode):
        self.X = np.append(sample, 1-sample)    # complement coding
        self.activateART()
        all_index = np.array(self.Y).argsort()[::-1]    # code competition
        # resonance check
        if all_index.size == 0:
            condition = False
        else:
            for i in range(self.category):
                resonanceIdx = all_index[i]
                condition = self.resonanceART(self.w[resonanceIdx])
                if condition:
                    break
        if condition:
            self.w[resonanceIdx] = self.updateART(self.w[resonanceIdx])
        else:
            weight = self.updateART(np.ones(sample.shape[1]*2))
            self.category += 1
            self.Y.append(1)
            if self.w is None:
                self.w = np.atleast_2d(weight)
            else:
                self.w = np.vstack((self.w, weight))

            self.maxPosNum.append(maxPosNum)
            self.numEvent.append(numEpisode)

    def updateART(self, weight):
        weight = (1 - self.lr) * weight + self.lr * (fuzzy_and(self.X, weight))
        return weight

    def resonanceART(self, weight):
        m = np.sum(fuzzy_and(self.X, weight)) / np.sum(self.X)
        if m >= self.rho:
            condition = True
        else:
            condition = False
        return condition

    def activateART(self):
        alpha = 1e-6
        for i in range(self.category):
            act_k = np.zeros((self.channel, 1))
            for j in range(self.channel):
                act_k[j] = self.gamma * np.sum(fuzzy_and(self.X, self.w[i])) / (alpha + np.sum(self.w[i]))
            self.Y[i] = np.sum(act_k)

    def train(self, X):
        X = np.array(X)
        for i in range(X.shape[0]):
            samples = np.array(X[i])
            sample, maxPosNum, numEpisode = self.seq_encoding(samples)
            self.zero_padding(sample)
            self.learningART(sample, maxPosNum, numEpisode)
        Y = self.testDeepART(X)
        return Y

    def testDeepART(self, X):
        X = np.array(X)
        oy = np.zeros((X.shape[0], np.array(X[0]).shape[1]))
        for i in range(X.shape[0]):
            samples = X[i]
            oy[i], _, _ = self.seq_encoding(np.array(samples))
        dataNum = oy.shape[0]
        Y = np.zeros((dataNum, len(self.Y)))
        for i in range(dataNum):
            self.X = np.hstack((oy[i], 1-oy[i]))
            self.activateART()
            ind = np.argmax(self.Y)
            YY = [0 if i != ind else x for i, x in enumerate(self.Y)]
            Y[i] = YY
        Y = np.array(Y)
        return Y

    def readout(self, Y):
        X = []
        if len(Y.shape) != 3:
            Y = np.array([Y])
        for i in range(Y.shape[0]):
            for j in range(Y[i].shape[0]):
                x = self.readoutDeepART(Y[i][j])
                if len(X) == 0:
                    X = np.array([x])
                else:
                    X = list(X)
                    X.append(x)
                    X = np.array(X)
        return X

    def readoutDeepART(self, Y):
        iw = 1
        bw = 2
        ow = 1
        ind = np.argmax(Y)
        output = np.zeros((self.numEvent[ind], self.w[ind][:self.w.shape[1]//2].shape[0]))
        for i in range(self.numEvent[ind]):
            if i  == 0:
                oy = self.w[ind][:self.w.shape[1]//2] * 10**self.maxPosNum[ind]
            else:
                oy = bx
            L = np.argmax(oy)
            for j in range(self.numEvent[ind]):
                if abs(iw * (bw * ow)**(j+1) - oy[L]) < 0.001:
                    E = j+1
                    break
                elif (iw * (bw * ow)**(j+1)) > oy[L]:
                    E = j
                    break
                elif j+1 == self.numEvent[ind]:
                    E = j+1
            ix = np.zeros((oy.shape[0]))
            ix[L] = 1
            bx = oy
            bx[L] = oy[L] - iw * (bw * ow)**E * ix[L]
            output[i] = ix
        return output


if __name__ == '__main__':
    x = [[[1.0, 0, 0, 0], [1.0, 0, 0, 0], [0, 1.0, 0, 0]], [[0, 0, 1.0, 0], [0, 0, 0, 1.0]]]
    x = np.array(x)
    model = deepART()
    model.train(x)
    Y = model.testDeepART(x)
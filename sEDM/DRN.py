import math
import numpy as np
import itertools
from utils import err, fuzzy_and, fuzzy_or

class DRN(object):
    def __init__(self, channel=1, lr=1.0, glr=1.0, alpha=1.0, rho=1, d=2, gamma=1):
        self.lr = lr  # learning rate
        self.glr = glr  # global learning rate
        self.alpha = alpha  # parameter for node activation
        self.rho = rho  # rho parameter
        self.d = d
        self.gamma = gamma
        self.X = []
        self.Y = []
        self.w = None  # weights for clusters (samples, weights)
        self.wg = None  # global weight vector
        self.n_category = 0  # number of categories
        self.group = np.array([])  # group container
        self.channel = channel  # number of channels in input

    def learningNN(self, point):       # sample = [-0.0213,-0.0578]
        self.X = point
        self.updateWg(self.X)
        self.activateNN()
        nu_index, all_index = self.code_comp()
        print(all_index)
        # resonance check
        if len(all_index) == 0:
            condition = False
        else:
            for i in range(self.n_category):
                resonanceIdx = all_index[i]
                condition = self.resonance(self.w[resonanceIdx])
                if condition:
                    break
        if condition:   # condition 추가
            self.w[resonanceIdx] = self.updateNN(point, self.w[resonanceIdx], self.lr)
        else:
            self.n_category += 1
            self.Y.append(1)
            if self.w is None:
                self.w = np.atleast_2d([np.hstack((self.X, self.X))])
            else:
                self.w = np.vstack((self.w, np.hstack((self.X, self.X))))
            nu_index = np.append(nu_index, self.n_category-1)     # np.append(nu_index, self.n_category)
        if self.n_category > 1:
            self.add_group(nu_index, condition, resonanceIdx)

    def add_group(self, index, condition, resonanceIdx):
        index = np.array(list(itertools.combinations(index, self.d)))
        for i in range(index.shape[0]):
            is_present = False
            for j in range(self.group.shape[0]):    # group = np.array([[1, 2,4, 3], [2, 3, 4,4],[4,5,6,4]]) shape:(3, 4)
                if all(self.group[j,1:3] == index[i]):
                    is_present = True
                    break
            if is_present == False:
                T = self.synapticStrength(index[i])
                if self.group.size == 0:
                    self.group = np.append(self.group, np.append(T, index[i]))
                    self.group = np.array([self.group])
                else:
                    self.group = np.vstack((self.group, np.append(T, index[i])))

        if condition == True:
            for j in range(self.group.shape[0]):
                if any(self.group[j,1:3] == resonanceIdx):
                    T = self.synapticStrength(self.group[j,1:3])
                    self.group[i, 0] = T

    def synapticStrength(self, index):
        p = []
        for i in range(index.size):
            for j in range(self.channel):
                points = np.vstack((self.w[int(index[i]), :self.w.shape[1]//2], self.w[int(index[i]), self.w.shape[1]//2:]))
                p.append(self.coM_calc(points))
        p = np.array(p)
        CoM_length = p[0] - p[1]
        T = math.exp(-self.alpha * np.linalg.norm(CoM_length, 2))
        return T

    def coM_calc(self, points):
        coM_point = np.sum(points, axis=0) / points.shape[0]
        return coM_point

    def updateWg(self, point):
        if self.n_category == 0:
            self.wg = np.append(self.X, self.X)
        else:
            e = err(self.X, self.wg)
            if e != 0:
                self.wg = self.updateNN(point, self.wg, self.glr)
                self.grouping()

    def updateNN(self, input, weight, lr):
        w1 = weight[:len(weight)//2]
        w2 = weight[len(weight)//2:]
        weight = (1 - lr) * weight + lr * (np.append(fuzzy_and(input, w1), fuzzy_or(input, w2)))
        return weight

    def activateNN(self):
        for i in range(self.n_category):
            e = err(self.X, self.w[i])
            self.Y[i] = math.exp(-self.alpha*e)

    def code_comp(self):
        all_index = np.array(self.Y).argsort()[::-1]
        nu_index = all_index
        if all_index.size == 0:     # if there is no node
            all_index = []
            nu_index = []
        elif all_index.size != 1:   # if there are several nodes
            if self.d > all_index.size:
                nu_index = all_index
            else:
                nu_index = all_index[:self.d]
            nu_index = np.sort(nu_index)
        return nu_index, all_index

    def resonance(self, weight):
        w1 = weight[:len(weight) // 2]
        w2 = weight[len(weight) // 2:]
        M = self.wg[len(self.wg) // 2:] - self.wg[:len(self.wg) // 2]
        S = fuzzy_or(self.X, w2) - fuzzy_and(self.X, w1)
        epsilon = 1e-6
        n = w1.size
        M[M == 0] = epsilon
        L = S / M
        if sum(L) / n <= 1 - self.rho:
            condition = True
        else:
            condition = False
        return condition

    def resonance_grouping(self, weight):
        w1 = weight[:len(weight) // 2]
        w2 = weight[len(weight) // 2:]
        M = self.wg[len(self.wg) // 2:] - self.wg[:len(self.wg) // 2]
        S = w2 - w1
        epsilon = 1e-6
        n = w1.size
        M[M == 0] = epsilon
        L = S / M
        if sum(L) / n <= 1 - self.rho:
            condition = True
        else:
            condition = False
        return condition

    def grouping(self):
        # Sort all activations in (nn.group)
        if self.group.size == 0:
            iter = self.group.size
        else:
            iter = np.array(self.group).shape[0]
        activation = np.zeros((iter))
        for i in range(iter):
            activation[i] = self.group[i, 0]
        sort_index = np.array(activation).argsort()[::-1]

        # Check resonance of multi_w
        delete_index = []
        for i in range(iter):
            # Group two target weights
            target_weight = self.group[sort_index[i], 1:3]
            if target_weight[0] == target_weight[1]:
                continue

            multi_w1 = fuzzy_and(self.w[int(target_weight[0]),:self.w.shape[1]//2], self.w[int(target_weight[1]),:self.w.shape[1]//2])
            multi_w2 = fuzzy_or(self.w[int(target_weight[0]),self.w.shape[1]//2:], self.w[int(target_weight[1]),self.w.shape[1]//2:])
            multi_w = np.append(multi_w1, multi_w2)

            condition = self.resonance_grouping(multi_w)

            if condition == True:    # grouping
                self.w[min(target_weight)] = multi_w
                # Substitute same indices
                for j in range(iter):
                    if any(self.group[j,1:3] == max(target_weight)):
                        ind = int(np.where(self.group[j,1:3] == max(target_weight))[0])
                        self.group[j,ind+1] = min(target_weight)
                delete_index.append(max(target_weight))     # Save the substituted node number
        # Remove all redundant nodes
        self.n_category = self.n_category - len(delete_index)
        delete_index = np.array(delete_index).argsort()[::-1]
        for i in range(delete_index.size):
            self.Y.pop(delete_index[i])
            self.w = np.delete(self.w, delete_index[i], axis=0)
        # Clear all groups
        for i in range(iter):
            if self.group[iter-i-1,1] == self.group[iter-i-1,2]:
                self.group = np.delete(self.group, iter-i+1, axis=0)
        # Clear same groups
        iter = self.group.shape[0]
        delete_group = []
        for i in range(self.group.shape[0]-1):
            for j in range(i+1, self.group.shape[0]):
                if all(self.group[i,1:3] == self.group[j,1:3]) or all(self.group[i,1:3] == np.append(self.group[j,2], self.group[j,1])):
                    if len(delete_group) == 0:
                        delete_group.append(j)
                    else:
                        if not any(np.array(delete_group) == j):
                            delete_group.append(j)
        delete_group = np.array(delete_group).argsort()[::-1]
        for i in range(delete_group.size):
            self.group = np.delete(self.group, delete_group[i], axis=0)
        # Match numbers in a group
        for j in range(delete_index.size):
            for i in range(self.group.shape[0]):
                if any(self.group[i, 1:3] > delete_index[j]):
                    ind = np.where(self.group[i, 1:3] > delete_index[j])[0]
                    for k in range(ind.size):
                        self.group[i, ind[k]+1] = self.group[i, ind[k]+1] - 1

        # Update all related lengths with updated weight
        for j in range(self.group.shape[0]):
            T = self.synapticStrength(self.group[j, 1:3])
            self.group[j][0] = T

    def train(self, X):
        X = np.array(X)
        for i in range(X.shape[0]):
            for j in range(np.array(X[i]).shape[0]):
                sample = np.array(X[i][j])
                self.learningNN(sample)
        Y = self.testDRN(X)
        return Y

    def testDRN(self, X):
        X = np.array(X)
        dataNum = X.shape[0]
        Y = [[] for _ in range(dataNum)]
        for i in range(dataNum):
            for j in range(np.array(X[i]).shape[0]):
                self.X = np.array(X[i][j])
                self.activateNN()
                ind = np.argmax(self.Y)
                YY = [0 if i != ind else x for i, x in enumerate(self.Y)]
                Y[i].append(YY)
        return Y

    def readout(self, Y):
        X = []
        XX = []
        for i in range(Y.shape[0]):
            for j in range(Y[i].shape[0]):
                ind = np.argmax(Y[i][j])
                if i == 0 and j == 0:
                    X.append([self.w[ind][:self.w.shape[1] // 2]])
                else:
                    X.append([self.w[ind][:self.w.shape[1]//2]])
            if i + 1 < Y.shape[0]:
                XX.append(np.array(X).squeeze(1))
                X = []
        XX.append(np.array(X).squeeze(1))
        XXX = np.array(XX)
        return XXX




if __name__ == '__main__':
    # x = np.array([np.array([[-0.0213,-0.0578],[-0.1272,0.0689],[-0.0897,-0.1516]]),
    #               np.array([[-0.0973,-0.0610],[ -0.0161,-0.0583]]),
    #               np.array([[0.0138,0.0760]]),
    #               np.array([[-0.1041,-0.1227],[0.0237,0.0132]]),
    #               np.array([[-0.0152,-0.0120],[0.0062,0.0617]])])
    x = np.array([np.array([[-0.0213, -0.0578], [-0.0213, -0.0578], [-0.0897, -0.1516]]),
                  np.array([[-0.0973,-0.0610],[ -0.0161,-0.0583]])])
    model = DRN()
    model.train(x)
    Y = model.testDRN(x)
    print(Y)

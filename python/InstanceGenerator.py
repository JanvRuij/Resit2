import numpy as np
import gurobipy as gp
from gurobipy import GRB


class SSP:
    def __init__(self, n, c):
        self.c = c
        self.n = n
        self.weights = np.random.randint(1, 180, self.n)

    def greedy(self, limit):
        U = np.copy(self.weights)
        I = np.ones(self.n)
        S = list()
        x = np.zeros((self.n, self.n))
        while np.count_nonzero(I) < limit:
            s = np.array([])
            max = np.argmax(U * I)
            print(max)
            while np.sum(s) + U[max] < self.c:
                s = np.append(s, U[max])
                I[max] = 0
                x[max, len(S)] = 1
                if np.count_nonzero(I) < limit:
                    break
                max = np.argmax(U * I)
            for index, weight in np.ndenumerate(U):
                if I[index] == 1:
                    if np.sum(s) + weight >= self.c:
                        s = np.append(s, weight)
                        I[index] = 0
                        x[index, len(S)] = 1
                        break
            S.append(s)
        return x

    def ILP_solver(self, preassigned):
        print(preassigned)
        model = gp.Model("ILP")
        model.setParam("TimeLimit", 5)
        x = model.addVars(self.n, self.n, vtype=GRB.BINARY, name="x")
        z = model.addVars(self.n, vtype=GRB.BINARY, name="z")
        model.addConstrs(gp.quicksum(self.weights[i] * x[i, k] for i in range(self.n)) >= self.c * z[k] for k in range(self.n))
        model.addConstrs(gp.quicksum(x[i, k] for k in range(self.n)) <= 1 for i in range(self.n))
        model.addConstrs(z[k] - z[k-1] <=  0 for k in range(2, self.n))
        model.setObjective(gp.quicksum(z[i] for i in range(self.n)), GRB.MAXIMIZE)
        for index in np.ndindex(preassigned.shape):
            i, j = index
            if preassigned[i][j] == 1:
                print((i,j))
                model.addConstr(x[j, i] == 1)

        model.optimize()

        if model.status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
            return model.ObjVal

    def Q_training(self):
        x = 4
        Q = [[[0.0 for _ in range(x)] for _ in range(x)] for _ in range(x)]
        nr_large = np.count_nonzero(self.weights > 170)
        nr_small = np.count_nonzero(self.weights < 10)
        a = np.random.randint(1, 4)
        if nr_large > 15:
            if nr_small > 10:
                Q[0][0][a] = self.ILP_solver(self.greedy(a * 10))
            else:
                Q[0][1][a] = self.ILP_solver(self.greedy(a * 10))
        elif nr_large > 10:
            if nr_small > 10:
                Q[1][0][a] = self.ILP_solver(self.greedy(a * 10))
            else:
                Q[1][1][a] = self.ILP_solver(self.greedy(a * 10))
        elif nr_large > 5:
            if nr_small > 10:
                Q[2][0][a] = self.ILP_solver(self.greedy(a * 10))
            else:
                Q[2][1][a] = self.ILP_solver(self.greedy(a * 10))
        else:
            if nr_small > 10:
                Q[3][0][a] = self.ILP_solver(self.greedy(a * 10))
            else:
                Q[3][1][a] = self.ILP_solver(self.greedy(a * 10))

        print(Q)


x = SSP(200, 200)
#x.greedy(0)
# x.ILP_solver(np.zeros((x.n, x.n)))
x.Q_training()

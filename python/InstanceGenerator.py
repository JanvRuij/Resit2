import numpy as np
import gurobipy as gp
from gurobipy import GRB


class SSP:
    def __init__(self, n, c):
        self.c = c
        self.n = n
        self.weights = np.random.randint(1, 180, self.n)

    def greedy(self):
        U = np.copy(self.weights)
        S = list()
        while len(U) > 0:
            s = np.array([])
            max = np.argmax(U)
            while np.sum(s) + U[max] < self.c:
                s = np.append(s, U[max])
                U = np.delete(U, max)
                if len(U) == 0:
                    break
                max = np.argmax(U)
            for index, weight in np.ndenumerate(U):
                if np.sum(s) + weight >= self.c:
                    s = np.append(s, weight)
                    U = np.delete(U, index)
                    break
            S.append(s)

    def ILP_solver(self):
        model = gp.Model("ILP")
        model.setParam("TimeLimit", 5)
        x = model.addVars(self.n, self.n, vtype=GRB.BINARY, name="x")
        z = model.addVars(self.n, vtype=GRB.BINARY, name="z")
        model.addConstrs(gp.quicksum(self.weights[i] * x[i, k] for i in range(self.n)) >= self.c * z[k] for k in range(self.n))
        model.addConstrs(gp.quicksum(x[i, k] for k in range(self.n)) <= 1 for i in range(self.n))
        model.addConstrs(z[k] - z[k-1] <=  0 for k in range(2, self.n))
        model.setObjective(gp.quicksum(z[i] for i in range(self.n)), GRB.MAXIMIZE)
        model.optimize()

        if model.status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
            pass




x = SSP(200, 200)
x.greedy()
x.ILP_solver()

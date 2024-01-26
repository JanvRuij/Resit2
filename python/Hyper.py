import numpy as np
from scipy.optimize import quadratic_assignment
import tqdm
import gurobipy as gp
from gurobipy import GRB, QuadExpr
from scipy.stats import binom


# Q parameters
gamma = 1
alpha = 0.1
epsilon = 0.1


class SSP:
    def __init__(self, n, c):
        self.c = c
        self.n = n
        self.weights = np.random.randint(1, 201, self.n)

    def reset(self):
        self.weights = np.random.randint(1, 201, self.n)

    def greedy(self, limit):
        # U is copy of the weights, I keeps track of indecices used
        U = np.copy(self.weights)
        I = np.ones(self.n)
        S = []
        x = np.zeros((self.n, self.n))
        while np.count_nonzero(I) > limit:
            s = np.array([])
            max = np.argmax(U * I)
            while np.sum(s) + U[max] < self.c:
                s = np.append(s, U[max])
                I[max] = 0
                x[max][len(S)] = 1
                if np.count_nonzero(I) <= limit:
                    break
                max = np.argmax(U * I)
            for index, weight in np.ndenumerate(np.sort(U)):
                if I[index] == 1:
                    if np.sum(s) + weight >= self.c:
                        s = np.append(s, weight)
                        I[index] = 0
                        x[index][len(S)] = 1
                        break
            S.append(s)

        return x, len(S)

    def ILP_solver(self, preassigned):
        # create the model
        model = gp.Model("ILP")
        model.setParam("OutputFlag", 0)
        model.setParam("TimeLimit", 5)

        # add vars
        x = model.addVars(self.n, self.n, vtype=GRB.BINARY, name="x")
        z = model.addVars(self.n, vtype=GRB.BINARY, name="z")

        # add constraints
        model.addConstrs(gp.quicksum(self.weights[i] * x[i, k] for i in range(self.n)) >= self.c * z[k] for k in range(self.n))
        model.addConstrs(gp.quicksum(x[i, k] for k in range(self.n)) <= 1 for i in range(self.n))
        model.addConstrs(z[k] - z[k-1] <= 0 for k in range(2, self.n))
        model.setObjective(gp.quicksum(z[i] for i in range(self.n)), GRB.MAXIMIZE)

        # set preassigned to 1
        indices = np.where(preassigned == 1)
        for i, j in zip(indices[0], indices[1]):
            model.addConstr(x[i, j] == 1)

        model.optimize()

        return model.ObjVal

    def Q_training(self):
        x = 4
        Q = [[[0.0 for _ in range(x)] for _ in range(x)] for _ in range(x)]
        N = [[[0.0 for _ in range(x)] for _ in range(x)] for _ in range(x)]
        average_list = []
        average_tracker = []
        idx1 = 0
        idx2 = 0
        quantiles = np.array([1.1, 0.75, 0.5, 0.25])
        for i in tqdm.tqdm((range(500))):
            if i % 10 == 0:
                print("Trained instance", i)
                average_list.append(sum(average_tracker)/10)
                average_tracker = []

            self.reset()

            nr_large = np.count_nonzero(self.weights > 150)
            nr_small = np.count_nonzero(self.weights < 50)

            prob_large = binom.cdf(nr_large, 200, 0.25)
            prob_small = binom.cdf(nr_small, 200, 0.25)
            for index, value in np.ndenumerate(quantiles):
                if prob_large < value:
                    idx1 = index[0]
                if prob_small < value:
                    idx2 = index[0]

            if np.random.random() < epsilon or Q[idx1][idx2][0] == Q[idx1][idx2][1]:
                a = np.random.randint(0, 4)
            else:
                a = Q[idx1][idx2].index(max(Q[idx1][idx2]))

            amount = 160 + a * 10
            x, _ = self.greedy(amount)
            r = self.ILP_solver(x)

            # Keep track of N
            N[idx1][idx2][a] += 1
            alpha = 1 / N[idx1][idx2][a]

            # Update the action value function
            Q[idx1][idx2][a] = Q[idx1][idx2][a] * (1 - alpha) + alpha * r
            average_tracker.append(r)

        print("average_tracker")
        print(average_tracker)

        return Q

    def Q_testing(self):
        quantiles = np.array([1.1, 0.75, 0.5, 0.25])
        nr_large = np.count_nonzero(self.weights > 190)
        nr_small = np.count_nonzero(self.weights < 10)
        prob_large = binom.cdf(nr_large, 200, 0.25)
        prob_small = binom.cdf(nr_small, 200, 0.25)
        for index, value in np.ndenumerate(quantiles):
            if prob_large < value:
                idx1 = index[0]
            if prob_small < value:
                idx2 = index[0]

        a = Q[idx1][idx2].index(max(Q[idx1][idx2]))
        amount = 160 + a * 10
        x, _ = self.greedy(amount)
        return self.ILP_solver(x)


x = SSP(200, 200)
Q = x.Q_training()
print(Q)
greedy = np.array([])
ILP = np.array([])
Q_testing = np.array([])

for i in range(100):
    x.reset()
    _, g_val = x.greedy(0)
    greedy = np.append(greedy, g_val)
    ILP = np.append(ILP, x.ILP_solver(np.zeros((x.n, x.n))))
    Q_testing = np.append(Q_testing, x.Q_testing())


print(f"Full gready: {np.average(greedy)}")
print(f"ILP Solver: {np.average(ILP)}")
print(f"Q testing: {np.average(Q_testing)}")

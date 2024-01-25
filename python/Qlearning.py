import numpy as np
import tqdm


# Q parameters
gamma = 1
alpha = 0.01
epsilon = 0.1


class BabyYahtzee:
    def __init__(self):
        # throw some dices
        self.dices = np.random.randint(1, 7, size=3)
        # keep track af what we start with
        self.dices_buffer = np.copy(self.dices)
        # start with 0 penalty
        self.penalty = 0

    def new_game(self):
        self.dices = np.random.randint(1, 7, size=3)
        self.dices_buffer = np.copy(self.dices)
        self.penalty = 0

    def reset(self):
        self.penalty = 0
        self.dices = np.copy(self.dices_buffer)

    def sort_dices(self):
        self.dices = np.sort(self.dices)

    def get_score(self):
        # calculate the score based on streets or combos
        self.dices = np.sort(self.dices)
        score = np.sum(self.dices)
        diffrence = np.diff(self.dices)
        is_street = np.all(diffrence == 1)
        highest_pair = np.count_nonzero(diffrence == 0)
        if highest_pair == 2:
            score += 60
        elif is_street:
            score += 30
        elif highest_pair == 1:
            score += 20
        return score - self.penalty

    def throw_dices(self, indices):
        if indices == []:
            return
        # throw some dices and receive penalties
        amount = len(indices)
        for i in indices:
            self.dices[i] = np.random.randint(1, 6)
        if amount == 3:
            self.penalty += 6
        elif amount == 2:
            self.penalty += 7
        else:
            self.penalty += 4

    # streets!
    def greedy1(self):
        # we ret
        self.dices = np.sort(self.dices)
        diffrence = np.diff(self.dices)

        # not going for a street with three or two of the same
        if np.sum(diffrence) <= 1:
            pass
        # if this is not the case we go for the streets!
        elif np.array_equal(diffrence, np.array([2, 1])):
            self.throw_dices([0])
        elif np.array_equal(diffrence, np.array([1, 2])):
            self.throw_dices([2])

    # comboosss
    def greedy2(self):
        self.dices = np.sort(self.dices)
        diffrence = np.diff(self.dices)

        # 3 the same!
        if np.sum(diffrence) <= 1:
            pass
        # first 2 are equal so rethrow last
        elif diffrence[0] == 0:
            self.throw_dices([2])
        # last 2 are euqal so rethrow first
        elif diffrence[1] == 0:
            self.throw_dices([0])

    def Q_training(self):
        global epsilon
        actions_taken = []
        new = 0
        # 8 action in each state
        action_space = [[0], [1], [2], [0, 1], [0, 2],
                        [1, 2], [0, 1, 2], []]
        Q = [[[[0.0 for _ in range(8)] for _ in range(6)] for _ in range(6)] for _ in range(6)]
        N = [[[[1 for _ in range(8)] for _ in range(6)] for _ in range(6)] for _ in range(6)]
        for _ in tqdm.tqdm((range(1000000))):
            # calculate the current score
            if actions_taken == []:
                start = self.get_score()
            else:
                start = new
            # substract one because computers count from 0
            first = self.dices[0] - 1
            second = self.dices[1] - 1
            third = self.dices[2] - 1
            if np.random.rand() < epsilon:
                # choose random action
                a = np.random.randint(0, 8)
                # take the random action
            else:
                a = Q[first][second][third].index(max(Q[first][second][third]))

            self.throw_dices(action_space[a])
            new = self.get_score()
            bonus = new - start
            actions_taken.append([a, [first, second, third], bonus])
            # if we are stopping, we start caclulating the reward for each stage
            if a == 7:
                # replay actions
                for av in actions_taken:
                    Q[av[1][0]][av[1][1]][av[1][2]][av[0]] = Q[av[1][0]][av[1][1]][av[1][2]][av[0]] * (1 - alpha) + alpha * av[2]
                self.new_game()
                actions_taken = []

        print(Q[0][0][0])
        print(Q[1][1][1])
        print(Q[1][2][4])
        return Q

    def Q_testing(self):
        action_space = [[0], [1], [2], [0, 1], [0, 2],
                        [1, 2], [0, 1, 2], []]
        a = 0
        self.sort_dices()
        for _ in range(100000):
            print(self.dices)
            # substract one because computers count from 0
            self.sort_dices()
            first = self.dices[0] - 1
            second = self.dices[1] - 1
            third = self.dices[2] - 1
            a = Q[first][second][third].index(max(Q[first][second][third]))
            print(f"Throw: {action_space[a]}")
            print(self.dices)
            self.throw_dices(action_space[a])
            if a == 7:
                print("done")
                return


x = BabyYahtzee()
greedy1 = np.array([])
greedy2 = np.array([])
greedy3 = np.array([])
Q_testing = np.array([])
Q = x.Q_training()
for i in range(100):
    x.new_game()
    # do just looking for streets
    x.greedy1()
    greedy1 = np.append(greedy1, x.get_score())
    x.reset()
    # look only for combinations
    x.greedy2()
    greedy2 = np.append(greedy2, x.get_score())
    x.reset()
    # look first for combinations and after that for streets
    x.greedy2()
    x.greedy1()
    greedy3 = np.append(greedy3, x.get_score())
    x.reset()
    x.Q_testing()
    Q_testing = np.append(Q_testing, x.get_score())



print(f"Only combos: {np.average(greedy2)}")
print(f"First combos then streets: {np.average(greedy3)}")
print(f"Only streets: {np.average(greedy1)}")
print(f"Q testing: {np.average(Q_testing)}")

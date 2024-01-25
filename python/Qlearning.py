import numpy as np
from numpy.lib import diff


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

        # goign for a street with 3 of the same is not smart..
        if np.sum(diffrence) == 0:
            pass
        # if this is not the case we go for the streets!
        elif np.array_equal(diffrence, np.array([2, 1])):
            self.throw_dices([0])
        elif np.array_equal(diffrence, np.array([1, 2])):
            self.throw_dices([2])

        return self.get_score()

    # comboosss
    def greedy2(self):
        self.dices = np.sort(self.dices)
        diffrence = np.diff(self.dices)

        # 3 the same!
        if np.sum(diffrence) == 0:
            pass
        # first 2 are equal so rethrow last
        elif diffrence[0] == 0:
            self.throw_dices([2])
        # last 2 are euqal
        elif diffrence[1] == 0:
            self.throw_dices([0])


x = BabyYahtzee()
for i in range(1):
    x.new_game()
    print(x.dices)
    greedy1 = x.greedy1()
    x.reset()
    print(x.dices)

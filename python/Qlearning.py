import numpy as np
import random as r


class BabyYahtzee:
    def __init__(self):
        self.dices = np.random.randint(1, 7, size=3)
        self.penalty = 0

    def get_score(self):
        self.dices = np.sort(self.dices)
        score = np.sum(self.dices)
        diffrence = np.diff(self.dices)
        is_sorted = np.all(diffrence == 1)
        highest_pair = np.count_nonzero(diffrence == 0)
        if highest_pair == 3:
            score += 60
        elif is_sorted:
            score += 30
        elif highest_pair == 2:
            score += 20
        return score - self.penalty

    def throw_dices(self, indices):
        amount = len(indices)
        for i in indices:
            self.dices[i] = r.randint(1, 6)
        if amount == 3:
            self.penalty += 6
        elif amount == 2:
            self.penalty += 7
        else:
            self.penalty += 4



x = BabyYahtzee()
x.throw_dices([1])
print(x.get_score())

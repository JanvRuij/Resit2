import numpy as np


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


x = BabyYahtzee()
greedy1 = np.array([])
greedy2 = np.array([])
greedy3 = np.array([])
for i in range(1000):
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

print(f"Only combos: {np.average(greedy2)}")
print(f"First combos then streets: {np.average(greedy3)}")
print(f"Only streets: {np.average(greedy1)}")

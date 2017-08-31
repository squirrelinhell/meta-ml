
import numpy as np
np.set_printoptions(precision=3, suppress=True)

from agents import Repeat, Softmax

def test():
    agent = Softmax(Repeat(
        [1, 2],
        [5, 4],
        ["10000.0", 10000]
    ))

    p = agent.action_batch([0.0] * 5)

    print(p)

test()

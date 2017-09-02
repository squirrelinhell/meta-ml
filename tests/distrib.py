
import timer

import mandalka
import numpy as np
np.set_printoptions(precision=3, suppress=True)

from agents import Repeat, Softmax, Gauss, Configure
from worlds import World

@mandalka.node
class TestWorld(World):
    def __init__(self):
        self.get_action_shape = lambda: (2,)
        self.get_reward_shape = lambda: (1,)

def test1():
    world = TestWorld()
    agent = Softmax(
        world,
        123,
        logits=Configure(
            Repeat,
            values=(
                [1.0, 2.0],
                [5.0, 4.0],
                [10000.0, 10000.0]
            )
        )
    )

    s, a = agent.step([None] * 5, [0.0] * 5)

    print(a)

def test2():
    world = TestWorld()
    agent = Gauss(world, 123)

    s, a = agent.step([None] * 3, [0.0] * 3)

    print(a)

test1()
test2()
assert timer.t() < 0.15

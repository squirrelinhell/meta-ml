
import timer

import mandalka
import numpy as np
np.set_printoptions(precision=3, suppress=True)

from agents import Configure, Cycle, Gauss, Softmax
from worlds import World

@mandalka.node
class TestWorld(World):
    def __init__(self, steps=5):
        self.get_action_shape = lambda: (2,)
        self.get_reward_shape = lambda: (1,)

        def trajectory(agent, seed):
            sta = [None]
            traj = []
            for _ in range(steps):
                sta, act = agent.step(sta, [0.0])
                traj.append((0.0, act[0], 0.0))
            return traj

        self.trajectory = trajectory

def test1():
    world = TestWorld()
    agent = Softmax(
        world,
        123,
        logits=Configure(
            Cycle,
            sequence=[
                [1.0, 2.0],
                [5.0, 4.0],
                [10000.0, 10000.0]
            ]
        )
    )

    for o, a, r in world.trajectory(agent, 0):
        print(a)

def test2():
    world = TestWorld(3)

    for o, a, r in world.trajectory(Gauss(world, 0), 0):
        print(a)

    for o, a, r in world.trajectory(Gauss(world, 1), 0):
        print(a)

test1()
test2()
assert timer.t() < 0.15

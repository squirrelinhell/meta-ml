
import mandalka
import numpy as np

from test_setup import timer
from agents import Cycle, Softmax
from values import Gauss
from worlds import World

@mandalka.node
class TestWorld(World):
    def __init__(self, steps=5):
        self.get_action_shape = lambda: (2,)
        self.get_reward_shape = lambda: (1,)

        def trajectory(agent):
            sta = [None]
            traj = []
            for _ in range(steps):
                sta, act = agent.step(sta, [0.0])
                traj.append((0.0, act[0], 0.0))
                print(act[0])
            return traj

        self.trajectories = (
            lambda agent, n: [trajectory(agent) for _ in range(n)]
        )

def test():
    world = TestWorld()
    agent = Softmax(
        world,
        0,
        logits=Cycle(
            sequence=[
                [1.0, 2.0],
                [5.0, 4.0],
                [10000.0, 10000.0]
            ]
        )
    )
    world.trajectories(agent, n=2)

test()
assert timer() < 0.02

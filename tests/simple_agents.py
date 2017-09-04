
import mandalka
import numpy as np
from agents import Cycle, Gauss, Softmax
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
                print(act[0])
            return traj

        self.trajectory = trajectory

def test1():
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
    world.trajectory(agent, 0)

def test2():
    world = TestWorld(3)
    world.trajectory(Gauss(world, 0), 0)
    world.trajectory(Gauss(world, 1), 0)

test1()
test2()
assert test_timer() < 0.15

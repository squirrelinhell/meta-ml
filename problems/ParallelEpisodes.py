
import mandalka

from . import Problem, Episode

@mandalka.node
class ParallelEpisodes(Problem):
    def __init__(self, problem):
        self.get_input_shape = problem.get_input_shape
        self.get_output_shape = problem.get_output_shape
        self.start_episode = lambda: PE_Episode(problem)

class PE_Episode(Episode):
    def __init__(self, problem):
        eps = []
        free_ids = []
        output_ids = []

        def next_input():
            if len(free_ids) < 1:
                eps.append(problem.start_episode())
                free_ids.append(len(eps)-1)

            i = free_ids.pop()
            while True:
                try:
                    result = eps[i].next_input()
                    break
                except StopIteration:
                    eps[i] = problem.start_episode()

            output_ids.append(i)
            return result

        def next_reward(output):
            nonlocal output_ids
            i, output_ids = output_ids[0], output_ids[1:]

            result = eps[i].next_reward(output)

            free_ids.append(i)
            return result

        self.next_input = next_input
        self.next_reward = next_reward

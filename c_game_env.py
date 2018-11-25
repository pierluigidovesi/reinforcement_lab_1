import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
from random import randint

walls = [[[0, 1], [0, 2]],
         [[1, 1], [1, 2]],
         [[2, 1], [2, 2]],
         [[1, 3], [1, 4]],
         [[2, 3], [2, 4]],
         [[1, 4], [2, 4]],
         [[1, 5], [2, 5]],
         [[3, 1], [4, 1]],
         [[3, 2], [4, 2]],
         [[3, 3], [4, 3]],
         [[3, 4], [4, 4]],
         [[4, 3], [4, 4]]]

maze_max = [5, 6]
goal = [4, 4]
horizon = 15
tot_episodes = 100000
precision = 1
discount = 1 - 1 / 30
mino_stand_still = False
verbose = 1

# init values
v_star = np.zeros((maze_max[0], maze_max[1], maze_max[0], maze_max[1]))
a_star = 10 * np.ones((maze_max[0], maze_max[1], maze_max[0], maze_max[1]))

class EnvAndPolicy:
    # init maze
    def __init__(self):
        self.walls = walls
        self.maze_max = maze_max
        self.goal = goal
        self.horizon = horizon
        self.v_star = v_star
        self.a_star = a_star

    # possible_actions
    def get_actions(self, state):
        agent_pos = state[0]
        assert agent_pos[0] < maze_max[0] and agent_pos[1] < maze_max[1]

        actions_list = []
        if self.reward(state) != 0:
            return actions_list

        move_north = [agent_pos, [agent_pos[0] - 1, agent_pos[1]]]
        move_south = [agent_pos, [agent_pos[0] + 1, agent_pos[1]]]
        move_east = [agent_pos, [agent_pos[0], agent_pos[1] + 1]]
        move_west = [agent_pos, [agent_pos[0], agent_pos[1] - 1]]

        if agent_pos[0] > 0 and not (move_north in walls) and not (move_north[::-1] in walls):
            actions_list.append(0)
            # print("north")
        if agent_pos[0] < maze_max[0] - 1 and not (move_south in walls) and not (move_south[::-1] in walls):
            actions_list.append(2)
            # print("south")
        if agent_pos[1] > 0 and not (move_west in walls) and not (move_west[::-1] in walls):
            actions_list.append(3)
            # print("west")
        if agent_pos[1] < maze_max[1] - 1 and not (move_east in walls) and not (move_east[::-1] in walls):
            actions_list.append(1)
            # print("east")
        actions_list.append(4)

        return actions_list

    def get_states_given_action(self, state, action):
        agent_pos = state[0]
        mino_pos = state[1]

        new_states = []

        if action == 0:
            new_agent_pos = [agent_pos[0] - 1, agent_pos[1]]
        elif action == 1:
            new_agent_pos = [agent_pos[0], agent_pos[1] + 1]
        elif action == 2:
            new_agent_pos = [agent_pos[0] + 1, agent_pos[1]]
        elif action == 3:
            new_agent_pos = [agent_pos[0], agent_pos[1] - 1]
        elif action == 4:
            new_agent_pos = agent_pos

        new_mino_pos = [mino_pos[0], mino_pos[1] + 1]
        if new_mino_pos[1] < maze_max[1]:
            new_states.append([new_agent_pos, new_mino_pos])

        new_mino_pos = [mino_pos[0], mino_pos[1] - 1]
        if new_mino_pos[1] >= 0:
            new_states.append([new_agent_pos, new_mino_pos])

        new_mino_pos = [mino_pos[0] + 1, mino_pos[1]]
        if new_mino_pos[0] < maze_max[0]:
            new_states.append([new_agent_pos, new_mino_pos])

        new_mino_pos = [mino_pos[0] - 1, mino_pos[1]]
        if new_mino_pos[0] >= 0:
            new_states.append([new_agent_pos, new_mino_pos])

        if mino_stand_still:
            new_states.append([new_agent_pos, mino_pos])

        return new_states

    def reward(self, state):
        if state[0] == state[1]:
            return -1
        elif state[0] == goal:
            return 1
        else:
            return 0

    def prob_given_action(self, new_states):
        return 1 / len(new_states)

    def get_index(self, state):
        index_state_time = copy.deepcopy(state)
        index_state_time = [item for sublist in index_state_time for item in sublist]
        return tuple(index_state_time)

    def fill_value_and_policy(self, state):
        matrix_index = self.get_index(state)
        value = np.array(np.ones(5) * -np.inf)
        possible_actions = self.get_actions(state)

        if self.reward(state) != 0:
            v_star[matrix_index] = self.reward(state)
        else:
            for action in possible_actions:
                value[action] = self.reward(state)
                possible_states = self.get_states_given_action(state, action)
                # print(possible_states)
                for next_state in possible_states:
                    next_state_index = self.get_index(next_state)
                    # print(next_state_index)
                    value[action] += discount * self.prob_given_action(possible_states) * v_star[next_state_index]

            v_star[matrix_index] = np.max(value)
            a_star[matrix_index] = np.argmax(value)
            #print(v_star)
            #print(a_star)

        return 0

    def main_loop(self, horizon=horizon):
        delta = 100
        n = 0
        while delta > precision*(1-discount)/discount:
        #while n != 10:
            delta = 100
            #delta = 0
            n += 1
            print(n)
            for x in range(maze_max[0]):
                for y in range(maze_max[1]):
                    for z in range(maze_max[0]):
                        for h in range(maze_max[1]):
                            state = [[x, y], [z, h]]
                            v_old = copy.deepcopy(v_star)
                            self.fill_value_and_policy(state)
                            #print(v_star-v_old)
                            #delta = np.sum(np.abs(v_star - v_old))
                            delta += LA.norm(v_star - v_old)
            delta = delta-100
            print(delta)

def get_movement_given_action(action, state):
    pos = copy.deepcopy(state)
    if action == 0:
        pos[0] -= 1
    elif action == 1:
        pos[1] += 1
    elif action == 2:
        pos[0] += 1
    elif action == 3:
        pos[1] -= 1
    return pos

#maze_map = np.zeros(maze_max)
def main():
    new_run = EnvAndPolicy()
    new_run.main_loop(15)
    print("Dynamic programming: done")
    distribution_array = np.zeros(horizon)

    step = 0
    state = [[0, 0], [4, 4]]
    for episode in tqdm(range(tot_episodes)):
        while state[0] != [4,4]:
            index = new_run.get_index(state)
            #print(index)
            action = a_star[index]

            # player movement
            state[0] = get_movement_given_action(action, state[0])

            # mino movement
            while True:
                mino_action = randint(0, 3)
                if mino_stand_still:
                    mino_action = randint(0, 4)
                nmp = get_movement_given_action(mino_action, state[1])
                #print(mino_action)
                if 0 < nmp[0] < maze_max[0] and 0 < nmp[1] < maze_max[1]:
                    state[1] = nmp
                    break

            step += 1

            if verbose:
                print(" _______________________", step)
                maze_map = np.zeros(maze_max)
                maze_map[tuple(state[0])] = 1
                maze_map[tuple(state[1])] = 2
                print(maze_map)
            # end while
        distribution_array[step] += 1
    plt.grid()
    plt.plot(distribution_array)
    plt.show()


if __name__ == "__main__":
    main()

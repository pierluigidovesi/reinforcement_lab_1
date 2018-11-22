import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

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
         [[3, 4], [4, 5]],
         [[4, 3], [4, 5]]]

maze_max = [4, 5]
goal = [4, 4]
horizon = 15
v_star = np.zeros((maze_max[0], maze_max[1], maze_max[0], maze_max[1], horizon))
a_star = 10 * np.ones((maze_max[0], maze_max[1], maze_max[0], maze_max[1], horizon))


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
        move_east  = [agent_pos, [agent_pos[0], agent_pos[1] + 1]]
        move_west = [agent_pos, [agent_pos[0], agent_pos[1] - 1]]

        if agent_pos[0] > 0 and not (move_north in walls) and not (move_north[::-1] in walls):
            actions_list.append(0)
            #print("north")
        if agent_pos[0] < maze_max[0] and not (move_south in walls) and not (move_south[::-1] in walls):
            actions_list.append(1)
            #print("south")
        if agent_pos[1] > 0 and not (move_west in walls) and not (move_west[::-1] in walls):
            actions_list.append(2)
            #print("west")
        if agent_pos[1] < maze_max[1] and not (move_east in walls) and not (move_east[::-1] in walls):
            actions_list.append(3)
            #print("east")
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

        new_mino_pos = [mino_pos[0], mino_pos[1]+1]
        if new_mino_pos[1]<=maze_max[1]:
            new_states.append([new_agent_pos, new_mino_pos])

        new_mino_pos = [mino_pos[0], mino_pos[1]-1]
        if new_mino_pos[1]>=0:
            new_states.append([new_agent_pos, new_mino_pos])

        new_mino_pos = [mino_pos[0]+1, mino_pos[1]]
        if new_mino_pos[0]<=maze_max[0]:
            new_states.append([new_agent_pos, new_mino_pos])

        new_mino_pos = [mino_pos[0]-1, mino_pos[1]]
        if new_mino_pos[0]>=0:
            new_states.append([new_agent_pos, new_mino_pos])

        return new_states

    def reward(self, state):
        if state[0] == state[1]:
            return -1
        elif state[0] == goal:
            return 1
        else:
            return 0

    def prob_given_action(self, new_states):
        return 1/len(new_states)

    def get_index(self, state, step):
        index_state_time = copy.deepcopy(state)
        index_state_time.append([step])
        index_state_time = [item for sublist in index_state_time for item in sublist]
        return tuple(index_state_time)

    def fill_value_and_policy(self, state, step):
        matrix_index = self.get_index(state, step)
        value = np.array(np.ones(5) * -np.inf)
        possible_actions = self.get_actions(state)

        if self.reward(state) != 0 or step == 14:
            v_star[matrix_index] = self.reward(state)
        else:
            for action in possible_actions:
                value[action] = self.reward(state)
                possible_states = self.get_states_given_action(state, action)
                for next_state in possible_states:
                    next_state_index = self.get_index(next_state, step+1)
                    print(next_state_index)
                    value[action] += self.prob_given_action(possible_states) * v_star[next_state_index]
            v_star[matrix_index] = np.max(value)
            a_star[matrix_index] = np.argmax(value)

        return 0

    def main_loop(self, horizon=horizon):
        for i in tqdm(range(horizon-1, -1, -1)):
            for x in range(maze_max[0]):
                for y in range(maze_max[1]):
                    for z in range(maze_max[0]):
                        for h in range(maze_max[1]):
                            state = [[x, y], [z, h]]
                            self.fill_value_and_policy(state, i)

    # plot maze of a state
    def plot_state(self,state):
        map = np.zeros((6, 5))
        # add walls
        print(np.array([item[0] for item in walls]))
        map[np.array([item[0] for item in walls])] = 1
        print(map)
        # add agent pos
        #map[state[0]] = 2
        # add mino pos
        #map[state[1]] = 3

        plt.pcolormesh(map)
        plt.axes().set_aspect('equal')  # set the x and y axes to the same scale
        plt.xticks([])  # remove the tick marks by setting to an empty list
        plt.yticks([])  # remove the tick marks by setting to an empty list
        plt.axes().invert_yaxis()  # invert the y-axis so the first row of data is at the top
        plt.show()

def main():
    new_run = EnvAndPolicy()
    new_run.main_loop(15)
    print("done")


if __name__ == "__main__":
    main()
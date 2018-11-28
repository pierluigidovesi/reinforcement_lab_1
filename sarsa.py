import numpy as np
import time
import copy
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import randint
from numpy import linalg as LA

maze_max = [4, 4]
goal = [1, 1] #bank
tot_episodes = 10000

discount = 0.8
epsilon = 0.1

# police is police
police_stand_still = False
verbose = 0
end_plot = 50
num_action = 5
number_steps = 10_000_000
playing_iter = 1000

# init values
initial_state = [[0, 0],[3,3]]
q_table = np.zeros((maze_max[0], maze_max[1], maze_max[0], maze_max[1], num_action))
n_table = np.zeros((maze_max[0], maze_max[1], maze_max[0], maze_max[1], num_action))
a_table = 10 * np.ones((maze_max[0], maze_max[1], maze_max[0], maze_max[1]),int)
sars_dataset = []

verbose = 1

class EnvAndPolicy:
    # init maze
    def __init__(self):
        self.maze_max = maze_max
        self.goal = goal
        self.q_table = q_table
        self.n_table = n_table


    # possible_actions
    def get_actions(self, state):
        agent_pos = state[0]

        assert agent_pos[0] < maze_max[0] or agent_pos[1] < maze_max[1]
        assert agent_pos[0] >= 0 or agent_pos[1] >= 0

        actions_list = []

        move_north = [agent_pos, [agent_pos[0] - 1, agent_pos[1]]]
        move_south = [agent_pos, [agent_pos[0] + 1, agent_pos[1]]]
        move_east = [agent_pos, [agent_pos[0], agent_pos[1] + 1]]
        move_west = [agent_pos, [agent_pos[0], agent_pos[1] - 1]]

        if agent_pos[0] > 0:
            actions_list.append(0)
            # print("north")
        if agent_pos[0] < maze_max[0] - 1:
            actions_list.append(2)
            # print("south")
        if agent_pos[1] > 0:
            actions_list.append(3)
            # print("west")
        if agent_pos[1] < maze_max[1] - 1:
            actions_list.append(1)
            # print("east")
        actions_list.append(4)

        return actions_list

    def get_states_given_action(self, state, action):
        agent_pos = state[0]
        police_pos = state[1]

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

        new_police_pos = [police_pos[0], police_pos[1] + 1]
        if new_police_pos[1] < maze_max[1]:
            new_states.append([new_agent_pos, new_police_pos])

        new_police_pos = [police_pos[0], police_pos[1] - 1]
        if new_police_pos[1] >= 0:
            new_states.append([new_agent_pos, new_police_pos])

        new_police_pos = [police_pos[0] + 1, police_pos[1]]
        if new_police_pos[0] < maze_max[0]:
            new_states.append([new_agent_pos, new_police_pos])

        new_police_pos = [police_pos[0] - 1, police_pos[1]]
        if new_police_pos[0] >= 0:
            new_states.append([new_agent_pos, new_police_pos])

        if police_stand_still:
            new_states.append([new_agent_pos, police_pos])

        return new_states

    def reward(self, state):
        if state[0] == state[1]:
            return -10
        elif state[0] == goal:
            return 1
        else:
            return 0

    def prob_given_action(self, new_states):
        return 1 / len(new_states)

    def get_index(self, state, action=10):

        index_state_time = copy.deepcopy(state)

        if action != 10:
            index_state_time.append([action])

        index_state_time = [item for sublist in index_state_time for item in sublist]

        return tuple(index_state_time)

    def sarsa(self, epsilon):

        next_state = initial_state
        possible_actions = self.get_actions(next_state)
        next_action = random.choice(possible_actions)

        for step in range(number_steps):

            state = copy.deepcopy(next_state)
            action = copy.deepcopy(next_action)

            print(action)

            reward = self.reward(state)

            next_possible_states = self.get_states_given_action(state, action)
            print(next_possible_states)
            next_state = random.choice(next_possible_states)

            #epsilon greedy
            random_num = random.uniform(0, 1)
            if random_num < epsilon:
                possible_actions = self.get_actions(next_state)
                next_action = random.choice(possible_actions)
            else:
                next_a_index = self.get_index(next_state)
                next_action = a_table[next_a_index]
                if next_action == 10:
                    possible_actions = self.get_actions(next_state)
                    next_action = random.choice(possible_actions)

            print(next_action)
            epsilon -= epsilon/number_steps
            matrix_index = self.get_index(state, action)

            n_table[matrix_index] += 1
            alpha = (1/n_table[matrix_index])**(2/3)

            next_matrix_index = self.get_index(next_state, next_action)
            q_next = q_table[next_matrix_index]
            q_now = q_table[matrix_index]

            q_table[matrix_index] += alpha*(reward + discount*(q_next-q_now))

            a_index = self.get_index(state)
            a_table[a_index] = np.argmax(q_table[a_index])

        return 0


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


# maze_map = np.zeros(maze_max)
def main():
    state = [[0, 0], [3, 3]]
    new_run = EnvAndPolicy()

    # offline on-policy training
    t_start = time.time()
    new_run.sarsa(epsilon)
    t_elapsed = time.time() - t_start
    print("Offline training time: ", t_elapsed)

    # plotting stuff
    total_reward = 0
    reward_list = []
    deriv_reward_list = []

    for step in range(playing_iter):
        index = new_run.get_index(state)
        #print(index)
        action = a_table[index]

        # player movement
        state[0] = get_movement_given_action(action, state[0])

        # police movement
        while True:
            police_action = randint(0, 3)
            if police_stand_still:
                police_action = randint(0, 4)
            nmp = get_movement_given_action(police_action, state[1])
            #print(police_action)
            if 0 < nmp[0] < maze_max[0] and 0 < nmp[1] < maze_max[1]:
                state[1] = nmp
                break

        # reward check
        total_reward += new_run.reward(state)
        reward_list.append(total_reward)
        deriv_reward_list.append(new_run.reward(state))

        if verbose:
            print(" _______________________", step, " money: ", total_reward)
            maze_map = np.zeros(maze_max)
            maze_map[tuple(state[0])] = 1
            maze_map[tuple(state[1])] = 2
            print(maze_map)
        # end while episode

    plt.grid()
    plt.plot(reward_list, label="total_reward")
    plt.plot(deriv_reward_list, label="step reward")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

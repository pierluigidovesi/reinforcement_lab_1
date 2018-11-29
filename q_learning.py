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

# police is police
police_stand_still = False
end_plot = 50
num_action = 5
dim_dataset = 100000
playing_iter = 100000
verbose = 0

# init values
q_table = np.zeros((maze_max[0], maze_max[1], maze_max[0], maze_max[1], num_action))
n_table = np.zeros((maze_max[0], maze_max[1], maze_max[0], maze_max[1], num_action))
a_table = np.zeros((maze_max[0], maze_max[1], maze_max[0], maze_max[1]))
sars_dataset = []
convergence_list = []


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
        assert agent_pos[0] < maze_max[0] and agent_pos[1] < maze_max[1]

        actions_list = []

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

    def fill_dataset(self, state, exploration_steps=1):

        for _ in tqdm(range(exploration_steps)):

            possible_actions = self.get_actions(state)
            random_action = random.choice(possible_actions)
            possible_states = self.get_states_given_action(state, random_action)
            random_state = random.choice(possible_states)
            reward_state = self.reward(state)

            sars_dataset.append([state, random_action, reward_state, random_state])

            state = random_state

        return [state, random_action, reward_state, random_state]

    def fill_q_table(self):

        for step in tqdm(sars_dataset):

            # fill placeholders
            state = step[0]
            action = step[1]
            reward = step[2]
            next_state = step[3]

            # update n_table
            matrix_index = self.get_index(state, action)
            a_index = self.get_index(state)
            n_table[matrix_index] += 1
            alpha = (1/n_table[matrix_index])**(2/3)

            # update q_table
            next_possible_actions = self.get_actions(next_state)
            q_next = np.array(np.ones(5) * -np.inf)
            #for i in range(4):
            for next_action in next_possible_actions:
                q_next[next_action] = q_table[self.get_index(next_state, next_action)]
            max_difference = np.max(q_next - q_table[matrix_index])
            q_table[matrix_index] += alpha*(reward + discount*max_difference)

            # update a_table
            best_action = np.argmax(q_table[a_index])
            possible_best_actions = self.get_actions(state)
            if best_action in possible_best_actions:
                a_table[a_index] = best_action
            else:
                a_table[a_index] = random.choice(possible_best_actions)

            # convergence plot
            convergence_list.append(np.mean(q_table))

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

    # init
    state = [[0, 0], [3, 3]]
    new_run = EnvAndPolicy()

    # dataset creation
    t_start = time.time()
    new_run.fill_dataset(state, dim_dataset)
    t_elapsed = time.time() - t_start
    # print("Dataset creation time: ", t_elapsed)

    # offline off-policy training
    t_start = time.time()
    new_run.fill_q_table()
    t_elapsed = time.time() - t_start
    # print("Offline training time: ", t_elapsed)

    # init plotting stuff
    total_reward = 0
    reward_list = []
    deriv_reward_list = []

    # main game loop
    for step in tqdm(range(playing_iter)):

        # get best action given state
        index = new_run.get_index(state)
        action = a_table[index]

        # perform player movement
        state[0] = get_movement_given_action(action, state[0])

        # perform police movement
        while True:
            police_action = randint(0, 3)
            if police_stand_still:
                police_action = randint(0, 4)
            nmp = get_movement_given_action(police_action, state[1])
            #print(police_action)
            if 0 < nmp[0] < maze_max[0] and 0 < nmp[1] < maze_max[1]:
                state[1] = nmp
                break

        # plot reward and check
        total_reward += new_run.reward(state)
        reward_list.append(total_reward)
        deriv_reward_list.append(new_run.reward(state))
        if new_run.reward(state) < 0:
            print(" DEAD! ")

        # verbose
        if verbose:
            print(" _______________________", step, " money: ", total_reward)
            maze_map = np.zeros(maze_max)
            maze_map[tuple(state[0])] = 1
            maze_map[tuple(state[1])] = 2
            print(maze_map)
    # end while episode

    # final plots and prints
    print("Derivative of reward given time: ", np.mean(deriv_reward_list))
    plt.grid()
    #plt.plot(reward_list, label="total_reward")
    #plt.plot(deriv_reward_list, label="step reward")
    #plt.legend()
    #plt.show()
    plt.plot(convergence_list, label="mean Q value")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

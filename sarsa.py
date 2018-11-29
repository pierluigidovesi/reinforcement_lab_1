import numpy as np
import time
import copy
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import randint
from numpy import linalg as LA

# map settings
maze_max = [4, 4]
goal = [1, 1]

# settings
discount = 0.8
epsilon_init = 0.1
epsilon_final = 0.1  # 1: pure exploration, 0: pure exploitation

police_stand_still = False
agent_actions = 5
train_iter = 100000
game_iter  = 100000

# init values
initial_state = [[0, 0], [3, 3]]
q_table = np.zeros((maze_max[0], maze_max[1], maze_max[0], maze_max[1], agent_actions))
n_table = np.zeros((maze_max[0], maze_max[1], maze_max[0], maze_max[1], agent_actions))
a_table = 100 * np.ones((maze_max[0], maze_max[1], maze_max[0], maze_max[1]), int)

# plot
smoothing = 0.001
train_reward_list = [0]
mean_q_table_list = []
verbose = 0


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

        # assert agent_pos[0] < maze_max[0] and agent_pos[1] < maze_max[1]
        # assert agent_pos[0] >= 0 and agent_pos[1] >= 0

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

        # assert agent_pos[0] < maze_max[0] and agent_pos[1] < maze_max[1]
        # assert agent_pos[0] >= 0 and agent_pos[1] >= 0

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

    def get_index(self, state, action=10):
        index_state_time = copy.deepcopy(state)
        # assert action != 100
        if action != 10:
            index_state_time.append([action])
        index_state_time = [item for sublist in index_state_time for item in sublist]
        return tuple(index_state_time)

    def sarsa(self):

        # init
        next_state = initial_state
        possible_actions = self.get_actions(next_state)
        next_action = random.choice(possible_actions)
        epsilon = epsilon_init

        for step in tqdm(range(train_iter)):

            # SARsa
            state = next_state
            action = next_action
            reward = self.reward(state)
            # sarSa
            next_possible_states = self.get_states_given_action(state, action)
            next_state = random.choice(next_possible_states)
            # sarsA: epsilon greedy (with init check) --> if rnd < epsilon: explore, else: exploit
            # exploit
            next_a_index = self.get_index(next_state)
            next_action = a_table[next_a_index]
            if random.uniform(0, 1) < epsilon or next_action == 100:
                # explore
                possible_actions = self.get_actions(next_state)
                next_action = random.choice(possible_actions)

            # moving epsilon linearly
            epsilon += (epsilon_final - epsilon_init)/train_iter

            # update n_table
            matrix_index = self.get_index(state, action)
            n_table[matrix_index] += 1
            alpha = (1/n_table[matrix_index])**(2/3)

            # update q_table
            next_matrix_index = self.get_index(next_state, next_action)
            q_next = q_table[next_matrix_index]
            q_now = q_table[matrix_index]
            q_table[matrix_index] += alpha*(reward + discount*(q_next-q_now))

            # update a_table with init check
            a_index = self.get_index(state)
            best_action = np.argmax(q_table[a_index])
            possible_best_actions = self.get_actions(state)
            if best_action in possible_best_actions:
                a_table[a_index] = best_action
            else:
                a_table[a_index] = random.choice(possible_best_actions)

            # convergence plot
            mean_q_table_list.append(np.mean(q_table))
            # reward plot
            train_reward_list.append(reward*smoothing + train_reward_list[-1]*(1 - smoothing))


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
    # t_start = time.time()
    new_run.sarsa()
    # t_elapsed = time.time() - t_start
    # print("Offline training time: ", t_elapsed)

    # plotting stuff
    total_reward = 0
    reward_list = []
    deriv_game_rewards_list = []

    # main game loop
    for step in tqdm(range(game_iter)):

        # init
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
            if 0 < nmp[0] < maze_max[0] and 0 < nmp[1] < maze_max[1]:
                state[1] = nmp
                break

        # reward check and plot
        total_reward += new_run.reward(state)
        reward_list.append(total_reward)
        deriv_game_rewards_list.append(new_run.reward(state))

        # verbose
        if verbose:
            print(" _______________________", step, " money: ", total_reward)
            maze_map = np.zeros(maze_max)
            maze_map[tuple(state[0])] = 1
            maze_map[tuple(state[1])] = 2
            print(maze_map)
        # end while episode

    # final plots and prints
    deriv_q_values_list = []
    for i in range(len(mean_q_table_list)-1):
        deriv_q_values_list.append(mean_q_table_list[i+1] - mean_q_table_list[i])

    print("Mean derivative of reward given time: ", np.mean(deriv_game_rewards_list))
    print("Mean derivative of Q values: ", np.mean(deriv_q_values_list))
    print("Last mean q_table: ", mean_q_table_list[-1])
    plt.grid()
    plt.suptitle("SARSA: epsilon from %f" % epsilon_init + " to %f" % epsilon_final + " (0 = pure exploit, 1 = pure explore)")
    plt.subplot(2,2,1)
    plt.title("Mean game reward per step = %f" %np.mean(deriv_game_rewards_list) + " (total game steps = %i)" % game_iter)
    plt.plot(reward_list, label="total game reward")
    plt.legend()

    # plt.plot(deriv_game_rewards_list, label="step reward")
    plt.subplot(2, 2, 2)
    plt.title("Last mean q_table: %f" % mean_q_table_list[-1] + " (total train steps = %i)" % train_iter)
    plt.plot(mean_q_table_list, label="mean Q_value")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.title("Last mean train reward: %f" %train_reward_list[-1])
    plt.plot(train_reward_list, label="train reward per step")
    # plt.plot(deriv_q_values_list, label="deriv Q value")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

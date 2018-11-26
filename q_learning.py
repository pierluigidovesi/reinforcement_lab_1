import numpy as np
import time
import copy
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
verbose = 0
end_plot = 50
num_action = 5

# init values
q_table = np.zeros((maze_max[0], maze_max[1], maze_max[0], maze_max[1], num_action))
n_table = np.zeros((maze_max[0], maze_max[1], maze_max[0], maze_max[1], num_action))


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
        if self.reward(state) != 0:
            return actions_list

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

    def get_index(self, state):
        index_state_time = copy.deepcopy(state)
        index_state_time = [item for sublist in index_state_time for item in sublist]
        return tuple(index_state_time)

    def fill_dataset(selfself):


    def fill_q_table(self, state):
        matrix_index = self.get_index(state)
        possible_actions = self.get_actions(state)


        for action in possible_actions:
            reward = self.reward(state)
            possible_states = self.get_states_given_action(state, action)
            # print(possible_states)
            #for next_state in possible_states:


                next_state_index = self.get_index(next_state)
                # print(next_state_index)
                value[action] += discount * self.prob_given_action(possible_states) * v_star[next_state_index]

            v_star[matrix_index] = np.max(value)
            a_star[matrix_index] = np.argmax(value)
            #print(v_star)
            #print(a_star)

        return 0

    def main_loop(self):
        delta = 100
        n = 0
        while delta > precision*(1-discount)/discount:
            delta = 0
            n += 1
            # print(n)
            for x in range(maze_max[0]):
                for y in range(maze_max[1]):
                    for z in range(maze_max[0]):
                        for h in range(maze_max[1]):
                            state = [[x, y], [z, h]]
                            v_old = copy.deepcopy(v_star)
                            self.fill_value_and_policy(state)
                            delta += LA.norm(v_star - v_old)
            # print(delta)
        print("number of value iterations: ", n, " final delta: ", delta, " < ", precision*(1-discount)/discount)

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
    t_start = time.time()
    new_run.main_loop()
    t_elapsed = time.time() - t_start
    print("Dynamic programming time: ", t_elapsed)
    distribution_array = np.zeros(end_plot)
    death_array = np.zeros(end_plot)
    win_array = np.zeros(end_plot)
    draw_array = np.zeros(end_plot)

    for episode in tqdm(range(tot_episodes)):
        step = 0
        state = [[0, 0], [4, 4]]
        while state[0] != [4, 4] and state[0] != state[1]:
            step += 1
            index = new_run.get_index(state)
            #print(index)
            action = a_star[index]

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

            if verbose:
                print(" _______________________", step)
                maze_map = np.zeros(maze_max)
                maze_map[tuple(state[0])] = 1
                maze_map[tuple(state[1])] = 2
                print(maze_map)
            # end while episode
        if step < end_plot:
            distribution_array[step] += 1
            if state[0] == state[1]:
                death_array[step] += 1
            if state[0] == goal:
                win_array[step] += 1
            if state[0] == state[1]:
                death_array[step] += 1
                print("DEAD!")
        else:
            draw_array[end_plot-1] += 1
    # end for 10000 episodes

    print("number of time out: ", np.sum(draw_array))
    plt.grid()
    plt.plot(distribution_array, label="distribution")
    plt.plot(death_array, label="death")
    plt.plot(win_array, label="win")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

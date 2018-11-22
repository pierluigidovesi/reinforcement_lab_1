import numpy as np
import matplotlib.pyplot as plt

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

#class maze
class Env:
    # init maze
    def __init__(self):
        self.walls = walls

    # possible_actions
    def get_actions(self, agent_pos):
        assert agent_pos[0] < maze_max[0] and agent_pos[1] < maze_max[1]
        actions_array = np.zeros(5)
        move_north = [agent_pos, [agent_pos[0] - 1, agent_pos[1]]]
        move_south = [agent_pos, [agent_pos[0] + 1, agent_pos[1]]]
        move_east  = [agent_pos, [agent_pos[0], agent_pos[1] + 1]]
        move_weast = [agent_pos, [agent_pos[0], agent_pos[1] - 1]]

        if agent_pos[0] > 0 and not (move_north in walls) and not (move_north[::-1] in walls):
            actions_array[0] = 1
            #print("north")
        if agent_pos[0] < maze_max[0] and not (move_south in walls) and not (move_south[::-1] in walls):
            actions_array[2] = 1
            #print("south")
        if agent_pos[1] > 0 and not (move_weast in walls) and not (move_weast[::-1] in walls):
            actions_array[3] = 1
            #print("weast")
        if agent_pos[1] < maze_max[1] and not (move_east in walls) and not (move_east[::-1] in walls):
            actions_array[1] = 1
            #print("east")
        actions_array[4] = 1

        return actions_array

    # possible_next_states
    def get_actions_and_states(self, agent_pos, mino_pos):

        actions_array = self.get_actions(agent_pos)
        new_states = []
        for i in range(np.size(actions_array)):
            if actions_array[i] == 1:
                if i == 0:
                    new_agent_pos = [agent_pos[0] - 1, agent_pos[1]]
                if i == 1:
                    new_agent_pos = [agent_pos[0], agent_pos[1] + 1]
                if i == 2:
                    new_agent_pos = [agent_pos[0] + 1, agent_pos[1]]
                if i == 3:
                    new_agent_pos = [agent_pos[0], agent_pos[1] - 1]
                if i == 4:
                    new_agent_pos = agent_pos

                new_mino_pos = [mino_pos[0], mino_pos[1]+1]
                new_states.append([new_agent_pos, new_mino_pos])

                new_mino_pos = [mino_pos[0], mino_pos[1]-1]
                new_states.append([new_agent_pos, new_mino_pos])

                new_mino_pos = [mino_pos[0]+1, mino_pos[1]]
                new_states.append([new_agent_pos, new_mino_pos])

                new_mino_pos = [mino_pos[0]-1, mino_pos[1]]
                new_states.append([new_agent_pos, new_mino_pos])

        return new_states

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



#!/usr/bin/env python3
from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import animation
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle
from random import randint

Colors = ['green', 'blue', 'orange', 'red','yellow','chocolate','darkorchid','hotpink','lightskyblue','palegreen','wheat','turquoise','navy','lightcoral','violet']
class Animation:
    def __init__(self, map, starts, goals, paths):
        self.map = map
        self.starts = starts
        # for start in starts:
        #     self.starts.append((start[1], len(self.map[0]) - 1 - start[0]))
        self.goals = goals
        # for goal in goals:
        #     self.goals.append((goal[1], len(self.map[0]) - 1 - goal[0]))

        self.paths = paths
        self.patches = []
        self.artists = []
        self.agents = dict()
        self.agent_names = dict()
        self.targets = dict()
        self.target_names = dict()
        self.T = 0
        # if paths:
        #     for path in paths:
        #         self.paths.append([])
        #         for loc in path:
        #             self.paths[-1].append((loc[1], len(self.map[0]) - 1 - loc[0]))

        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
        #figure settings
        handles, labels = self.figure_settings(self.ax)
        legend = plt.legend(handles, labels, bbox_to_anchor=(1, 1), loc="upper left", framealpha=1)

        # plot obstacles and map
        cmap = matplotlib.colors.ListedColormap(['white', 'black'])
        self.ax.pcolormesh(self.map, cmap=cmap, alpha=1.0, edgecolors='black')
        #print(self.paths)

        for i in range(int(len(self.goals))):

            name = str(i)
            self.targets[i] = Rectangle((self.goals[i][1], self.goals[i][0]), 1.0, 1.0, facecolor='grey')
            self.targets[i].original_face_color = 'grey'
            self.patches.append(self.targets[i])
            # target_text_flag = True
            # text_offset = 0.0
            # if target_text_flag:
            #     self.ax.text(self.goals[i][1] + text_offset,
            #                  self.goals[i][0] + text_offset,
            #                  "T" + str(i), fontweight="bold", color="red")
            self.target_names[i] = self.ax.text(goals[i][1] - 0.5, goals[i][0] -0.5, name, color='red')
            self.target_names[i].set_horizontalalignment('center')
            self.target_names[i].set_verticalalignment('center')
            self.artists.append(self.target_names[i])


        for i in range(len(self.paths)):
            name = str(i)
            self.agents[i] = Circle((self.starts[i][1], self.starts[i][0]), 0.5, facecolor=Colors[i % len(Colors)],
                                    edgecolor='black')
            self.agents[i].original_face_color = Colors[i % len(Colors)]
            self.patches.append(self.agents[i])
            self.T = max(self.T, len(paths[i]) - 1)
            self.agent_names[i] = self.ax.text(starts[i][1], starts[i][0] + 0.25, name, fontweight='bold')
            self.agent_names[i].set_horizontalalignment('center')
            self.agent_names[i].set_verticalalignment('center')
            self.artists.append(self.agent_names[i])


        self.animation = animation.FuncAnimation(self.fig, self.animate_func,
                                                 init_func=self.init_func,
                                                 frames=int(self.T + 1) * 10,
                                                 interval=100,
                                                 blit=True)



        ## plot targets
        # for idx_target in range(int(len(self.goals))):
        #
        #     self.ax.add_patch(Rectangle((self.goals[idx_target][1], self.goals[idx_target][0]), 1.0, 1.0,facecolor='grey'))
        #     # ax.scatter(targets_position[2 * idx_target] + 0.5, targets_position[2 * idx_target + 1] + 0.5,
        #     # marker="x", color="red")
        #     target_text_flag = True
        #     text_offset = 0.0
        #     if target_text_flag:
        #         self.ax.text(self.goals[idx_target][1] + text_offset,
        #                 self.goals[idx_target][0] + text_offset,
        #                 "T" + str(idx_target), fontweight="bold", color="red")

        #self.animation = animation.FuncAnimation(self.fig, self.animate_func, init_func=self.init_func, frames=20, interval = 100, blit = True)


    def figure_settings(self, ax):
        """
        Settings for the figure.

        cluster_legend_flag = True if plot cluster legends
        path_legend_flag = True if plot path legends
        """
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.set_xlim([0, len(self.map)])
        ax.set_ylim([0, len(self.map[0])])

        # set legends

        colors = ["blue"]
        marker_list = ["o"]
        labels = ["Agent"]
        f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none", markersize=12, markerfacecolor="None",
         markeredgecolor='black')[0]
        handles = [f(marker_list[i], colors[i]) for i in range(len(labels))]
        handles.append(patches.Patch(color="black", alpha=0.5))
        handles.append(patches.Patch(color="black", alpha=1))
        labels.extend(["Task", "Obstacles"])
                # a tuple includes the handles and labels of legend
        return handles, labels

    def animate_func(self, t):
        for k in range(len(self.paths)):
            pos = self.get_state(t / 10, self.paths[k])
            self.agents[k].center = (pos[1]+ 0.5, pos[0]+ 0.5)
            self.agent_names[k].set_position((pos[1] + 0.5, pos[0] + 0.5))

        for j in range(int(len(self.goals))):
            self.targets[j].center = (self.goals[j][1] + 0.5, self.goals[j][0] + 0.5)
            self.target_names[j].set_position((self.goals[j][1] + 0.5, self.goals[j][0] + 0.5))

        # reset all colors
        for _, agent in self.agents.items():
            agent.set_facecolor(agent.original_face_color)
        # for _, target in self.targets.items():
        #     target.set_facecolor(target.original_face_color)
        targets_array = [target for _, target in self.targets.items()]
        # check drive-drive collisions
        agents_array = [agent for _, agent in self.agents.items()]
        for i in range(0, len(agents_array)):
            for j in range(i + 1, len(agents_array)):
                d1 = agents_array[i]
                d2 = agents_array[j]
                pos1 = np.array(d1.center)
                pos2 = np.array(d2.center)
                if np.linalg.norm(pos1 - pos2) < 0.7:
                    d1.set_facecolor('red')
                    d2.set_facecolor('red')
                    print("COLLISION! (agent-agent) ({}, {}) at time {}".format(i, j, t / 10))


        for i in range(0, len(agents_array)):
            for j in range(int(len(self.goals))):
                d1 = agents_array[i]
                pos1 = np.array(d1.center)
                d2 = targets_array[j]
                pos2 = np.array(d2.center)
                if np.linalg.norm(pos1 - pos2) < 0.2:
                    d2.set_facecolor('white')

        return self.patches + self.artists



    def init_func(self):
        for p in self.patches:
            self.ax.add_patch(p)
        for a in self.artists:
            self.ax.add_artist(a)
        return self.patches + self.artists
        pass

    @staticmethod
    def show():
        plt.show()

    @staticmethod
    def get_state(t, path):
        if int(t) <= 0:
            return np.array(path[0])
        elif int(t) >= len(path):
            return np.array(path[-1])
        else:
            pos_last = np.array(path[int(t) - 1])
            pos_next = np.array(path[int(t)])
            pos = (pos_next - pos_last) * (t - int(t)) + pos_last
            return pos




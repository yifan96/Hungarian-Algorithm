
import numpy as np
#
# a_origin = [[300, 250, 180, 320, 270, 190, 220, 260],
#             [290, 310, 190, 180, 210, 200, 300, 190],
#             [280, 290, 300, 190, 190, 220, 230, 260],
#             [290, 300, 190, 240, 250, 190, 180, 210],
#             [210, 200, 180, 170, 160, 140, 160, 180]]
# a = [[80, 40, 0, 150, 30, 40, 50, 80],
#      [60, 90, 0, 0, 30, 40, 120, 0],
#      [40, 60, 100, 0, 0, 50, 40, 60],
#      [60, 80, 0, 60, 70, 30, 0, 20],
#      [0, 0, 10, 10, 0, 0, 0, 10]]
# a = [[80, 40, 0, 140, 100, 40, 50, 70],
#      [70, 100, 10, 0, 40, 50, 130, 0],
#      [50, 70, 110, 0, 10, 60, 50, 60],
#      [60, 80, 0, 50, 70, 30, 0, 10],
#      [0,  0, 10, 0, 0, 0, 0, 0]]
#
# def unbalanced_assignment(a):
#     a = np.array(a)
#     zero_mat = (a == 0)
#     print(zero_mat)
#
#     marked_zero = []
#     zero_bool_mat_copy = zero_mat.copy()
#     marked_zero_row = []
#     marked_zero_col = []
#
#     for i in range(len(marked_zero)):
#         marked_zero_row.append(marked_zero[i][0])
#         marked_zero_col.append(marked_zero[i][1])
#     # Recording possible answer positions by marked_zero
#     while len(marked_zero) < len(a[0]):
#         for row in range(len(zero_bool_mat_copy)):
#             if np.sum(zero_bool_mat_copy[row] == True) == 1:
#                 marked_zero.append([row, np.where(zero_bool_mat_copy[row] == True)[0][0]])
#                 zero_bool_mat_copy[:, np.where(zero_bool_mat_copy[row] == True)[0][0]] = [False for i in range(
#                     len(zero_bool_mat_copy))]
#                 # zero_bool_mat_copy[row, :] = [False for i in range(len(zero_bool_mat_copy[0]))]
#                 # break
#                 # print("change 1")
#                 # print(zero_bool_mat_copy)
#                 continue
#                 # if np.sum()
#         # print(marked_zero)
#         # Now the left
#         left_zero_row = np.where(zero_bool_mat_copy == True)[0]
#         left_zero_col = np.where(zero_bool_mat_copy == True)[1]
#         # print(len(np.where(zero_bool_mat_copy==True)[0]))
#         min_zero_index = [-1, -1]
#         min_zero = np.Inf
#         index = 0
#         if len(marked_zero) < len(a[0]):
#             for i in range(len(np.where(zero_bool_mat_copy == True)[0])):
#                 if a[left_zero_row[i]][left_zero_col[i]] < min_zero:
#                     min_zero_index = [left_zero_row[i], left_zero_col[i]]
#                     min_zero = a[left_zero_row[i]][left_zero_col[i]]
#                     index = i
#             marked_zero.append(min_zero_index)
#             if len(left_zero_col) > 0:
#                 zero_bool_mat_copy[:, left_zero_col[index]] = [False for i in range(len(zero_bool_mat_copy))]
#             # print(zero_bool_mat_copy)
#             # print(marked_zero)
#         # continue
#
#     print("marked zero")
#     print(marked_zero)
#
#
# if __name__ == '__main__':
#     a = [[80, 40, 0, 150, 30, 40, 50, 80],
#          [60, 90, 0, 0, 30, 40, 120, 0],
#          [40, 60, 100, 0, 0, 50, 40, 60],
#          [60, 80, 0, 60, 70, 30, 0, 20],
#          [0, 0, 10, 10, 0, 0, 0, 10]]
#     a1 = [[80, 40, 0, 140, 100, 40, 50, 70],
#          [70, 100, 10, 0, 40, 50, 130, 0],
#          [50, 70, 110, 0, 10, 60, 50, 60],
#          [60, 80, 0, 50, 70, 30, 0, 10],
#          [0, 0, 10, 0, 0, 0, 0, 0]]
#     unbalanced_assignment(a)
#     unbalanced_assignment(a1)
# from random import randint
# num_obs = 10
# obs_left_top_corner = list()
# map_width = 5
# map_height = 5
# size_obs_width = 1
# size_obs_height = 3
# value_non_obs = 0
# value_obs = 255
# map_array = np.array([value_non_obs] * (map_width * map_height)).reshape(-1, map_width)
# #map_array = np.array([[1,2,4,5],
#                       [7,8,9,10],
#                       [11,12,13,14],
#                       [0,0,0,0]])
# print(map_array[1:2][:])
# print(map_array[1:2][:,0:2])
# print(map_array[1:3][0])
# print(map_array)
#
# if num_obs > 0:
#     for idx in range(num_obs):
#         obs_left_top_corner.append([randint(1, map_width - size_obs_width - 1),
#                                     randint(1, map_height - size_obs_height - 1)])
# #X[:,  m:n]即取矩阵X的所有行中的的第m到n-1列数据，含左不含右。
#         obs_mat = map_array[obs_left_top_corner[idx][1]:obs_left_top_corner[idx][1] + size_obs_height][:, obs_left_top_corner[idx][0]:obs_left_top_corner[idx][0] + size_obs_width]
#         # print(obs_mat.shape)
#         # print(obs_left_top_corner)
#         # print(obs_mat)
#
#         map_array[obs_left_top_corner[idx][1]:obs_left_top_corner[idx][1] + size_obs_height][:, obs_left_top_corner[idx][0]:obs_left_top_corner[idx][0] + size_obs_width] = value_obs * np.ones(obs_mat.shape)
# print(obs_left_top_corner)
# print(map_array)
# def check_target_collide_agents(self, target_position: list, agent_positions: list):
#     """
#     Check if a target collides with all the agents.
#     Return True if a collision exists.
#     """
#     collision_flag = False
#     for idx in range(0, len(agent_positions), 2):
#         collision_flag = collision_flag or (target_position[0] == agent_positions[idx]) and (
#                     target_position[1] == agent_positions[idx + 1])
#     return collision_flag
#
# targets = list()
# num_targets = 5
# while len(targets) < num_targets:
#     goal = [randint(1, map_width - 1), randint(1, map_width - 1)]
#     while (map_array[goal[1]][goal[0]] != value_non_obs) or (check_target_collide_agents(goal, agents)):
#         goal = [randint(1, map_width - 1), randint(1, map_width - 1)]
#     targets.extend(goal)

# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
#
# a = np.array([1,1,0,0,0,0,0,0,1,0,0,0]).reshape(3,-1)
#
# print(len(a))
#
# _, ax = plt.subplots(1,1)
# cmap = matplotlib.colors.ListedColormap(['white', 'black'])
# # # map_array = MySimulator.map_array
# # # print(map_array)
# ax.pcolormesh(a, cmap=cmap, edgecolors='none')
#
# # plt.show()
#
# goals = [1,2,3,4,5,6,7,8]
# goals = np.array(goals).reshape(-1,2)
# print(goals)
# goal_1 = [tuple(x) for x in goals]
# print(goal_1)
# goal_2 = [tuple(reversed(x)) for x in goals]
# print(goal_2)

# import heapq
# a = [(1, (7, 12), {'loc': (7, 12), 'cost': 1}),
#      (1, (8, 11), {'loc': (8, 11), 'cost': 1}),
#      (1, (8, 13), {'loc': (8, 13), 'cost': 1}),
#      (1, (9, 12), {'loc': (9, 12), 'cost': 1})]
# print(heapq.heappop(a))
# print(heapq.heappop(a))
# print(heapq.heappop(a))
# print(heapq.heappop(a))
# a= [[(1,2),(3,4)],[(5,6),(7,8)]]

# a=[[1,2,1,2],[3,4,3,4],[5,6,5,6],[7,8,7,8],[9,10,9,10],[11,12,11,12]]
# n=3
# for i in range(n):
#     print(i)
#     for col_idx in range(i+1, len(a)):
#         a[col_idx][i]=0
# print(a)
# #
# import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib import animation
# import time
# fig, ax =plt.subplots()
# x = np.arange(0, 2 * np.pi, 0.01)
# line, = ax.plot(x, np.sin(x))
#
# def animate():
#      # print(i)
#      # line.set_ydata(np.sin(x+i/10))
#      # return line,
#      line.set_ydata(x)
#      return line,
#
#
# def init():
#      line.set_ydata(x)
#      return line,
#
# ani = animation.FuncAnimation(fig=fig, func=animate, frames=10000, init_func=init, interval=20)
#
# plt.show()



# from vrp import VRP_solver
# a =  [
#         [
#             0, 548, 776, 696, 582, 274, 502, 194, 308, 194, 536, 502, 388, 354,
#             468, 776, 662
#         ],
#         [
#             548, 0, 684, 308, 194, 502, 730, 354, 696, 742, 1084, 594, 480, 674,
#             1016, 868, 1210
#         ],
#         [
#             776, 684, 0, 992, 878, 502, 274, 810, 468, 742, 400, 1278, 1164,
#             1130, 788, 1552, 754
#         ],
#         [
#             696, 308, 992, 0, 114, 650, 878, 502, 844, 890, 1232, 514, 628, 822,
#             1164, 560, 1358
#         ],
#         [
#             582, 194, 878, 114, 0, 536, 764, 388, 730, 776, 1118, 400, 514, 708,
#             1050, 674, 1244
#         ],
#         [
#             274, 502, 502, 650, 536, 0, 228, 308, 194, 240, 582, 776, 662, 628,
#             514, 1050, 708
#         ],
#         [
#             502, 730, 274, 878, 764, 228, 0, 536, 194, 468, 354, 1004, 890, 856,
#             514, 1278, 480
#         ],
#         [
#             194, 354, 810, 502, 388, 308, 536, 0, 342, 388, 730, 468, 354, 320,
#             662, 742, 856
#         ],
#         [
#             308, 696, 468, 844, 730, 194, 194, 342, 0, 274, 388, 810, 696, 662,
#             320, 1084, 514
#         ],
#         [
#             194, 742, 742, 890, 776, 240, 468, 388, 274, 0, 342, 536, 422, 388,
#             274, 810, 468
#         ],
#         [
#             536, 1084, 400, 1232, 1118, 582, 354, 730, 388, 342, 0, 878, 764,
#             730, 388, 1152, 354
#         ],
#         [
#             502, 594, 1278, 514, 400, 776, 1004, 468, 810, 536, 878, 0, 114,
#             308, 650, 274, 844
#         ],
#         [
#             388, 480, 1164, 628, 514, 662, 890, 354, 696, 422, 764, 114, 0, 194,
#             536, 388, 730
#         ],
#         [
#             354, 674, 1130, 822, 708, 628, 856, 320, 662, 388, 730, 308, 194, 0,
#             342, 422, 536
#         ],
#         [
#             468, 1016, 788, 1164, 1050, 514, 514, 662, 320, 274, 388, 650, 536,
#             342, 0, 764, 194
#         ],
#         [
#             776, 868, 1552, 560, 674, 1050, 1278, 742, 1084, 810, 1152, 274,
#             388, 422, 764, 0, 798
#         ],
#         [
#             662, 1210, 754, 1358, 1244, 708, 480, 856, 514, 468, 354, 844, 730,
#             536, 194, 798, 0
#         ],
#     ]
# depot =[0,0,0,0]
# num_agent = 4
# solver = VRP_solver(a,num_agent,depot)
# solver.solve()


# a = [[] for i in range(3)]
# print(a)
# for i in range(3):
#     for j in range(2):
#         a[i].append(j)
# print(a)
a = [[1,0,1,2,9,14,1,3], [31,1,4,51,11,0,1] ,[0,2,3,5,7], [1,1,6,7,8],[4,5],[7,7,7,7,7],[10,2]]
#
# print (min(a, key=len))
# # [1, 1, 6, 7, 8]
# print(a)
# for x in a:
#     x.pop()
# print(a)
# a = [[(x +1) for x in row] for row in a]
# print(a)
# print(sum([1,2,3,4,5]))
#
# print([0]*4)
# a = []
# z = 6
# for i in range(6):
#     a.append([i])
# print(a)
# def find_min_list(list):
#     list_len = [len(i) for i in list]
#     print(list_len)
#     min_len = min(list_len)
#     key = [j for j, x in enumerate(a) if x == min_len]
#     print(key)

#print output#
# find_min_list(a)
# task_to_do = [0,0,0,0]
# list_len = [len(i) for i in a]
# list_len = np.array(list_len)
# print(list_len)
# print(list_len.min())
# print(type(list_len[1]))
# x = np.where(list_len ==list_len.min())[0]
# print(x)
# for i in range(len(x)):
#     list_len[x[i]] = np.Inf
# print(list_len)
done = [4,6]
res = [ele for idx, ele in enumerate(a) if idx not in done]
print(a)
print(res)

# a = [[1,2,4],[1,2]]
# b = 10
# if len(a[0])<b:
#     a[0].extend([a[0][-1]]*(b-len(a[0])))
# print(a)
# done = [[1,2],[2,3]]
# if done:
#     print('x')
# if np.array(done).size>0:
#     print('ddd')
#
# a=[[1,2],[2,4]]
# b=[[]]*2
# print(b)
# print([a[i]+b[i] for i in range(len(a))])
# paths_temp = [[(14, 19),(1,2)], [(9, 3), (8, 3), (7, 3), (6, 3), (5, 3), (4, 3), (3, 3), (2, 3)], [(6, 15), (7, 15), (8, 15)], [(1, 15), (1, 16), (1, 17)]]
#
# path = [[]] *4
# paths_temp = [i[1 : ] for i in paths_temp]
# print(paths_temp)
# print([path[i] + paths_temp[i] for i in range(4)])
start= [(4, 19), (15, 6), (20, 4), (11, 3)]
path = [[start[i]] for i in range(4)]
print(path)

path1 = [[1,2],[3,4],[]]
countlist = path1.count([])
print(countlist)
countlist = start.count((4,19))
print(countlist)
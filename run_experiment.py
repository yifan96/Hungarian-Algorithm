# MSPF and MCPF libraries
import math
import time
import os, sys

sys.path.append(os.path.abspath('/home/yifan/PycharmProjects/pythonProject/libmcpf'))
sys.path.append(os.path.abspath('/home/yifan/PycharmProjects/pythonProject/pytspbridge'))
print(sys.path)
import cbss_msmp
import cbss_mcpf
import common as cm
import SA_makespan
import SA_total_cost

import numpy as np
from Environment import Environment
import matplotlib
import matplotlib.pyplot as plt
import random
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle
import time as timer
from Visualize import Animation
from cbs import CBSSolver
from Build_assignment_matrix import Build_assignment_matrix_vrp
from Build_assignment_matrix import Build_assignment_matrix_mspf
from vrp import VRP_solver


def run_experiment(seed):
    map_height_meter = 32
    map_width_meter = 32
    map_resolution = 1
    value_non_obs = 0
    value_obs = 1
    # create a simulator
    MySimulator = Environment(map_height_meter, map_width_meter, map_resolution, value_non_obs, value_obs)
    # number of obstacles
    num_obs = 41
    # [width, length] size of each obstacle [meter]
    size_obs = [1,10]

    ###############################
    # change map and positions of agents and tasks here!!!
    random.seed(seed)
    ###############################

    # generate random obstacles
    MySimulator.generate_random_obs(num_obs, size_obs)
    # randomly generate agents and targets
    num_agents = 5
    num_targets = 5
    # agents position and targets position are for visualization
    # reversed
    agents_position, targets_position = MySimulator.generate_agents_and_targets(num_agents, num_targets)
    print("agents positions are:")
    print(agents_position)
    print("goal positions are:")
    print(targets_position)
    map_array = MySimulator.map_array
    print(map_array)

    #
    # build assignment matrix
    # change agents and targets position to list of tumples
    # starts and ends are for path planning in array of the map
    # agents position and targets position are for visualization, should be reversed for path planner
    starts = [tuple(reversed(x)) for x in np.array(agents_position).reshape(-1, 2)]
    ends = [tuple(reversed(x)) for x in np.array(targets_position).reshape(-1, 2)]

    ###############################
    # A_star from CBS, extremely slow
    ###############################
    # timer_a_start = timer.time()
    # Assignment = Build_assignment_matrix_vrp(map_array, starts, ends)
    # cost_matrix = Assignment.build_matrix()
    # print('time1')
    # print(timer.time()-timer_a_start)
    # mat = np.matrix(cost_matrix)
    # with open('outfile.txt', 'wb') as f:
    #     for line in mat:
    #         np.savetxt(f, line, fmt='%.2f', delimiter=',')
    #
    # with open('outfile_cplex.txt', 'w') as out_file:
    #     with open('outfile.txt', 'r') as in_file:
    #         for line in in_file:
    #             out_file.write('['+ line.rstrip('\n') + ']' + '\n')

    ###############################
    # Normal A_star
    ###############################
    starts_idx = [node2idx(i, map_width_meter + 1) for i in starts]
    ends_idx = [node2idx(i, map_width_meter + 1) for i in ends]
    timer_a_start1 = timer.time()
    Assignment_mspf = Build_assignment_matrix_mspf(map_array,starts_idx,ends_idx)
    cost_matrix = Assignment_mspf.build_matrix()
    # mat = np.matrix(cost_matrix)
    # with open('outfile.txt', 'wb') as f:
    #     for line in mat:
    #         np.savetxt(f, line, fmt='%.2f', delimiter=',')
    #
    # with open('outfile_cplex.txt', 'w') as out_file:
    #     with open('outfile.txt', 'r') as in_file:
    #         for line in in_file:
    #             out_file.write('['+ line.rstrip('\n') + ']' + '\n')
    if len(cost_matrix) == 0:
         return 99999,99999,99999,99999, 99999



    # print('timer2')
    # print(timer.time()-timer_a_start1)
    #####################################################
    ###########     COST MATRIX     #####################
    #####################################################
    print("the cost matrix is:")
    print(cost_matrix)
    # print("the cost matrix is:")
    # print(cost_matrix_mspf)
    print('starts:')
    print(starts)
    print('ends:')
    print(ends)

    starts_visual = [tuple(reversed(x)) for x in np.array(agents_position).reshape(-1, 2)]
    ends_visual = [tuple(reversed(x)) for x in np.array(targets_position).reshape(-1, 2)]


    # ################################
    # ### Google or-tools ############
    # ################################
    # start_time_routing = timer.time()
    # vrp_solver = VRP_solver(cost_matrix,num_agents,list(range(0,num_agents)))
    # assignment_result, max_route_distance, total_distance = vrp_solver.solve()
    # time_routing = timer.time() - start_time_routing
    # print("time_routing",time_routing)
    # #####################################################
    # ###########     ASSIGNMENT RESULT     ###############
    # #####################################################
    # print('assingment result in list form is')
    # print(assignment_result)
    # print('Maximum of the route distances: {}m'.format(max_route_distance))
    # print('Total route distance', total_distance)



    ###################################
    ##### Yu's greedy + linear SA  ####
    ###################################
    # assignment_result, max_cost, total_cost, time_routing = SA_makespan.simulated_annealing(num_agents,num_targets,cost_matrix)
    # print('assingment result in list form is')
    # print(assignment_result)
    # print('time routing',time_routing)
    # print('max cost', max_cost)
    # print('total cost',total_cost)

    assignment_result, max_cost, total_cost, time_routing = SA_makespan.simulated_annealing(num_agents, num_targets, cost_matrix)
    print('assingment result in list form is')
    print(assignment_result)
    print('time routing', time_routing)
    print('max cost', max_cost)
    print('total cost', total_cost)

   #  # Draw agents and targets in the map
   #  _, ax = plt.subplots(1,1)
   #  cmap = matplotlib.colors.ListedColormap(['white', 'black'])
   #  # plot the map
   #  ax.pcolormesh(map_array, cmap=cmap, alpha=1.0, edgecolors='black')
   #  MySimulator.plot_agents(agents_position, 0.0, ax)
   #  MySimulator.plot_targets(targets_position, [], [], 0.0, ax)
   #  plt.show()
   # #




    #################### Iteratively use CBS
    # task index to be finished
    start_time_CBS = timer.time()
    task_to_do = [0] * num_agents
    curr_loc = starts
    paths = [[starts[i]] for i in range(num_agents)]
    while True : #sum(task_to_do) <= num_targets - num_agents :
        curr_targets = []
        curr_targets_index = []
        for i in range(num_agents):
            if len(assignment_result[i]):
                 curr_targets.append(ends[assignment_result[i][task_to_do[i]]])
            else:
                curr_targets.append(starts[i])
            #curr_targets_index.append(assignment_result[i][task_to_do[i]])
        # print("curr loc")
        # print(curr_loc)
        # print("curr targets")
        # print(curr_targets)
        # print('curr_loc')
        # print(curr_loc)
        # print('curr_targets')
        # print(curr_targets)
        # print('curr target index')
        # print(curr_targets_index)

        cbs = CBSSolver(map_array, curr_loc, curr_targets)
        #cbs = CBSSolver(map_array, curr_loc, ends[0:4])
        paths_temp = cbs.find_solution()
        # find the shortest path in paths_temp, crop paths_temp, update curr_loc and curr_targets
        paths_temp_len = [len(i) for i in paths_temp]
        paths_temp_len = np.array(paths_temp_len)
        bool_assignment = [len(i)>0 for i in assignment_result]
        # print('bool assignment')
        # print(bool_assignment)
        redundant = num_agents - sum(bool_assignment)
        if any(item is False for item in bool_assignment):
            idx_agent_done = np.append(np.where(task_to_do == np.array([len(i) - 1 for i in assignment_result]))[0],
                                       np.array([i for i, x in enumerate(assignment_result) if not x]))
        else:
            idx_agent_done = np.where(task_to_do == np.array([len(i) - 1 for i in assignment_result]))[0]

        # if len(assignment_result[i]):
        #     idx_agent_done = np.where(task_to_do == np.array([len(i) - 1 for i in assignment_result]))[0]
        # else:
        #     idx_agent_done = np.append(np.where(task_to_do == np.array([len(i) - 1 for i in assignment_result]))[0],
        #                                np.array([i for i, x in enumerate(assignment_result) if not x]))
        # else:

        #else:
        # print('task to do')
        # print(task_to_do)
        # print(np.array([len(i)-1 for i in assignment_result]))
        # print(type(paths_temp_len))
        # print('idx_agent_done')
        # print(idx_agent_done)

        if idx_agent_done.size == 0:
            min_path_len = paths_temp_len.min()
            min_path_idx = np.where(paths_temp_len == paths_temp_len.min())[0]
            curr_loc = [i[min_path_len-1] for i in paths_temp]
            # print('path temp')
            # print(paths_temp)
            # print('path temp len')
            # print(paths_temp_len)
            # print("min task len")
            # print(min_path_len)
            # print('min path idx')
            # print(min_path_idx)
            # print('task to do idx')
            # print(task_to_do)

            # visit next assigned task if the index not exceeds boudary.
            for idx_to_next_task in range(len(min_path_idx)):
                if task_to_do[min_path_idx[idx_to_next_task]] < len(assignment_result[min_path_idx[idx_to_next_task]])-1:
                    task_to_do[min_path_idx[idx_to_next_task]] += 1
            paths_temp = [i[1 : min_path_len] for i in paths_temp]
            paths = [paths[i] + paths_temp[i] for i in range(num_agents)]
            # print('paths')
            # print(paths)

        if idx_agent_done.size > 0 and idx_agent_done.size < num_agents:
            # delete path of agent that has finished its last assigned task and then find the minimum one
            res = np.array([ele for idx, ele in enumerate(paths_temp_len) if idx not in idx_agent_done])
            # print(res)
            min_path_len = res.min()
            min_path_idx = np.where(paths_temp_len == res.min())[0]
            # crop paths, if
            # print(' done path temp')
            # print(paths_temp)
            # print('done path temp len')
            # print(paths_temp_len)
            # print(" done min task len")
            # print(min_path_len)
            # print('done min path idx')
            # print(min_path_idx)

            #patch the done agent
            for idx in idx_agent_done:
                while len(paths_temp[idx]) < min_path_len:
                    paths_temp[idx].extend([paths_temp[idx][-1]])
            curr_loc = [i[min_path_len-1] for i in paths_temp]
            for idx_to_next_task in range(len(min_path_idx)):
                if task_to_do[min_path_idx[idx_to_next_task]] < len(assignment_result[min_path_idx[idx_to_next_task]])-1:
                    task_to_do[min_path_idx[idx_to_next_task]] += 1
            paths_temp = [i[1 : min_path_len] for i in paths_temp]
            paths = [paths[i] + paths_temp[i] for i in range(num_agents)]
            # print('paths')
            # print(paths)

        elif idx_agent_done.size == num_agents:
            paths_temp = [i[1:] for i in paths_temp]
            paths = [paths[i] + paths_temp[i] for i in range(num_agents)]
            break

    # for every route in path, from back to start, keep only one if repeated (calculate total cost)
    print(paths)
    for i in range(len(paths)):
        while paths[i][-1] == paths[i][-2] and len(paths[i]) > 2:
            paths[i] = paths[i][:-1]
        if len(paths[i])==2:
            paths[i] = paths[i][:-1]
    print(paths)

    time_CBS = timer.time() - start_time_CBS

    # max_route_distance_CBS = max(len(a) for a in paths)
    # print('Maximum of the route distances after CBS: {}m'.format(max_route_distance_CBS))

    print('time for routing', time_routing)
    print('time for CBS', time_CBS)

    #####################################################
    ###########      ANIMATION     ######################
    #####################################################
   # here starts and ends are for planning, reversed back in Class animation
    # paths coordinates are in planning manner, reversed in animation func
    print('SA-reCBS Path', paths)
    sa_recbs_total = sum(len(i) for i in paths)
    print('SA-reCBS total cost',sa_recbs_total)

    paths = [[i] for i in starts]
    animation = Animation(map_array, starts_visual , ends_visual, paths)
    animation.show()



    ##############################################################
    ######################## MSPF ################################
    ##############################################################
    configs = dict()
    configs["problem_str"] = "msmp"
    configs["mtsp_fea_check"] = 1
    configs["mtsp_atLeastOnce"] = 1
    # this determines whether the k-best TSP step will visit each node for at least once or exact once.
    configs["tsp_exe"] = "/home/yifan/PycharmProjects/pythonProject/pytspbridge/tsp_solver/LKH-2.0.9/LKH"
    configs["time_limit"] = 60
    configs["eps"] = 9999
    starts_idx = [node2idx(i, map_width_meter+1) for i in starts]
    targets_idx = [node2idx(i,map_width_meter+1) for i in ends]
    res_dict, path = cbss_msmp.RunCbssMSMP(map_array, starts_idx, targets_idx, targets_idx, configs)

    print(res_dict)
    path = [item[0:-1] for item in path]
    print(path)

    if len(path) == 0:
        max_cost_mspf = math.inf
        total_cost_mspf = math.inf
        total_time_mspf = math.inf
    else:
        max_cost_mspf = max(len(i) for i in path)
        total_cost_mspf = sum(len(i) for i in path)
        total_time_mspf = res_dict["n_tsp_time"] + res_dict["search_time"]
    print("maximum cost of MSMP", max_cost_mspf)
    print("total cost of MSMP", total_cost_mspf)
    print("total time of MSPF", total_time_mspf)
    # animation = Animation(map_array, starts_visual, ends_visual, path)
    # animation.show()
    return time_routing, time_CBS, sa_recbs_total , total_cost_mspf, total_time_mspf





# transform node to index. in 2x2 map, (0,0) is indexed as 0, (1,1) is indexed as 3
def node2idx(node, map_width):
    idx = map_width * node[0] + node[1]
    return idx



# def run_CBSS_MSMP():
#     """
#     fully anonymous case, no assignment constraints.
#     """
#     print("------run_CBSS_MSMP------")
#     ny = 100
#     nx = 100
#     grids = np.zeros((ny, nx))
#     grids[5, 3:7] = 1  # obstacles
#     print(grids)
#
#     starts = [11, 22, 33, 88, 99, 444, 443, 232, 43, 34]
#     targets = [40, 38, 27, 66, 72, 81, 83, 123, 343, 23, 555, 342, 456, 778, 997, 331, 1, 233, 400, 329]
#     # dests = [19,28,37,46,69                       ,223,654,232,31,76]
#     dests = [11, 22, 33, 88, 99, 444, 443, 232, 43, 34]



    # return

if __name__ == "__main__":
    time_route_list = []
    time_CBS_list = []
    sa_recbs_total_list = []
    total_cost_mspf_list = []
    total_time_mspf_list = []
    res = random.sample(range(-99999, 9999), 200)
    print('random int list')
    print(res)
    for i in range(1):
        print('current seed')
        print(res[i])
        time_routing, time_CBS, sa_recbs_total,_,_= run_experiment(res[i]) # seed
        time_route_list.append(time_routing)
        time_CBS_list.append(time_CBS)
        sa_recbs_total_list.append(sa_recbs_total)
        # total_cost_mspf_list.append(total_cost_mspf)
        # total_time_mspf_list.append(total_time_mspf)
    print(time_route_list)
    print(time_CBS_list)
    # with open('Routing_20_10.txt', 'w') as f:
    #     for element in time_route_list:
    #         if element != 99999:
    #            f.write(f"{element}\n")
    #
    # with open('CBS_20_10.txt', 'w') as f:
    #     for element in time_CBS_list:
    #         if element != 99999:
    #             f.write(f"{element}\n")
    #
    # with open('sarecbs_totalcost_20_10', 'w') as f:
    #     for element in sa_recbs_total_list:
    #         if element != 99999:
    #             f.write(f"{element}\n")

    # with open('total_cost_mspf_20_10.txt', 'w') as f:
    #     for element in total_cost_mspf_list:
    #         if element != 99999:
    #             f.write(f"{element}\n")
    #
    # with open('total_time_mspf_20_10.txt', 'w') as f:
    #     for element in total_time_mspf_list:
    #         if element != 99999:
    #             f.write(f"{element}\n")





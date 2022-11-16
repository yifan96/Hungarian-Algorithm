from single_agent_planner import compute_heuristics, a_star, get_sum_of_cost
import timeit
import numpy as np

class Build_assignment_matrix_unbalanced_hungarian(object):
    def __init__(self, map, curr_locs, goals):
        self.map = map
        self.curr_locs = curr_locs
        self.goals = goals
    def build_matrix(self):
        cost_matrix = np.ones(shape = [len(self.curr_locs), len(self.goals)]) * np.Inf
        for idx_goal in range(len(self.goals)):
            heuristics = compute_heuristics(self.map, self.goals[idx_goal])
            for idx_agent in range(len(self.curr_locs)):
                path = a_star(self.map, self.curr_locs[idx_agent], self.goals[idx_goal], heuristics, 1, [])
                cost_matrix[idx_agent][idx_goal] = len(path) -1
        return cost_matrix

class Build_assignment_matrix_vrp(object):
    def __init__(self, map, starts, goals):
        # starts and goals are lists of tuples
        self.map = map
        self.starts = starts
        self.goals = goals
        #self.nodes = starts + goals
        self.nodes = starts + goals
    def build_matrix(self):
        cost_matrix = np.ones(shape = [len(self.nodes), len(self.nodes)]) * np.Inf
        print(self.nodes)
        for idx_node in range(len(self.nodes)):
            heuristics = compute_heuristics(self.map, self.nodes[idx_node])
            for idx_node_ in range(len(self.nodes)):
                path = a_star(self.map, self.nodes[idx_node], self.nodes[idx_node_], heuristics, 1, [])
                cost_matrix[idx_node][idx_node_] = len(path) -1
        for i in range(len(self.starts)):
            for col_idx in range(i + 1, len(cost_matrix)):
                cost_matrix[col_idx][i] = 0
        return cost_matrix

class Build_assignment_matrix_vrp_manhaton(object):
    def __init__(self, map, starts, goals):
        # starts and goals are lists of tuples
        self.map = map
        self.starts = starts
        self.goals = goals
        self.nodes = starts + goals
    def build_matrix(self):
        cost_matrix = np.ones(shape = [len(self.nodes), len(self.nodes)]) * np.Inf
        print(self.nodes)
        for idx_node in range(len(self.nodes)):
            for idx_node_ in range(len(self.nodes)):
                cost_matrix[idx_node][idx_node_] = abs(self.nodes[idx_node][0]-self.nodes[idx_node_][0])+abs(self.nodes[idx_node][1]-self.nodes[idx_node_][1])
        for i in range(len(self.starts)):
            for col_idx in range(i + 1, len(cost_matrix)):
                cost_matrix[col_idx][i] = 0
        return cost_matrix



class Build_assignment_matrix_vrp_cplex(object):
    def __init__(self, map, starts, goals):
        # starts and goals are lists of tuples
        self.map = map
        self.starts = starts
        self.goals = goals
        #self.nodes = starts + goals
        self.nodes = goals + starts + starts
    def build_matrix(self):
        cost_matrix = np.zeros(shape = [len(self.nodes), len(self.nodes)])
        print(self.nodes)
        for idx_node in range(len(self.goals)+len(self.starts)):
            heuristics = compute_heuristics(self.map, self.nodes[idx_node])
            for idx_node_ in range(len(self.goals)):
                path = a_star(self.map, self.nodes[idx_node], self.nodes[idx_node_], heuristics, 1, [])
                cost_matrix[idx_node][idx_node_] = len(path) -1
        return cost_matrix




def main():

    maze = [[0 ,0, 0, 0 ,0 ,0 ,0 ,0 ,0, 0, 0 ,0 ,0, 0, 0, 0 ,0 ,0 ,0 ,0 ,0],
            [0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0, 0 ,0 ,0 ,0, 0 ,0 ,0, 0 ,0],
            [0, 0 ,0 ,0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0 ,0, 0, 0 ,0 ,0 ,0 ,0 ,0, 0, 0 ,0 ,0, 0, 0, 0 ,0 ,0 ,0 ,0 ,0],
            [0 ,0, 0, 0 ,0 ,0 ,0 ,0 ,0, 0, 0 ,0 ,0, 0, 0, 0 ,0 ,0 ,0 ,0 ,0],
            [0 ,0, 0, 0 ,0 ,0 ,0 ,0 ,0, 0, 0 ,0 ,0, 0, 0, 0 ,0 ,0 ,0 ,0 ,0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0 ,0, 0, 0 ,0 ,0 ,0 ,0 ,0, 0, 0 ,0 ,0, 0, 0, 0 ,0 ,0 ,0 ,0 ,0],
            [0 ,0, 0, 0 ,0 ,0 ,0 ,0 ,0, 0, 0 ,0 ,0, 0, 0, 0 ,0 ,0 ,0 ,0 ,0],
            [0 ,0, 0, 0 ,0 ,0 ,0 ,0 ,0, 0, 0 ,0 ,0, 0, 0, 0 ,0 ,0 ,0 ,0 ,0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0 ,0, 0, 0 ,0 ,0 ,0 ,0 ,0, 0, 0 ,0 ,0, 0, 0, 0 ,0 ,0 ,0 ,0 ,0],
            [0 ,0, 0, 0 ,0 ,0 ,0 ,0 ,0, 0, 0 ,0 ,0, 0, 0, 0 ,0 ,0 ,0 ,0 ,0],
            [0 ,0, 0, 0 ,0 ,0 ,0 ,0 ,0, 0, 0 ,0 ,0, 0, 0, 0 ,0 ,0 ,0 ,0 ,0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0 ,0, 0, 0 ,0 ,0 ,0 ,0 ,0, 0, 0 ,0 ,0, 0, 0, 0 ,0 ,0 ,0 ,0 ,0],
            [0 ,0, 0, 0 ,0 ,0 ,0 ,0 ,0, 0, 0 ,0 ,0, 0, 0, 0 ,0 ,0 ,0 ,0 ,0]]

    start = [(3,14)]
    end = [(8,12)]
    print(maze[3][14])
    assignment = Build_assignment_matrix_unbalanced_hungarian(maze,start,end)
    #compute_heuristics(maze, end)
    matrix = assignment.build_matrix()
    print(matrix)



if __name__ == '__main__':
    main()
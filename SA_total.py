import numpy as np
import time

# filename = 'outfile.txt'

# this function relocate the position of each customer and find the move with lowest cost increment
def relocate(solution,cost,cuslist,data):
    bestcostinc = 10000
    for i in cuslist:
        for j in range(0,len(solution)):
            if i in solution[j]:
               pos = solution[j].index(i)
               costinc = data[solution[j][pos-1],i] + data[i,solution[j][pos+1]] \
                          - data[solution[j][pos-1],solution[j][pos+1]]
        for j in range(0, len(solution)):
            for k in range(1, len(solution[j])):
                costinc += data[solution[j][k - 1], i] + data[i, solution[j][k]] \
                          - data[solution[j][k - 1], solution[j][k]]
                if costinc < bestcostinc:
                    bestcostinc = costinc
                    bestveh = j
                    bestpos = k
                    bestcus = i
    for j in range(0, len(solution)):
        if bestcus in solution[j]:
            pos = solution[j].index(bestcus)
            cost[j] -= data[solution[j][pos - 1], bestcus] + data[bestcus, solution[j][pos + 1]] \
                       - data[solution[j][pos - 1], solution[j][pos + 1]]
            solution[j].remove(bestcus)
    solution[bestveh].insert(bestpos, bestcus)
    cost[bestveh] += bestcostinc

def simulated_annealing(num_agents, num_tasks, cost_matrix):

    start = time.time()

# load file
    data = cost_matrix

# k is the number of vehicles, c is the number of customers
    K = num_agents
    C = num_tasks
    N = K + C

# initialize customerlist, solution and cost vectors
    solution = []
    cost = []
    for i in range(0,K):
        solution.append([i,i])
        cost.append(0)

    cuslist = []
    for i in range(K,N):
        cuslist.append(i)

# main loop for greedy insertion, in each loop we insert the customer with lowest cost increment to
# the whole solution
    while len(cuslist) != 0:
        bestcostinc = 10000
        for i in cuslist:
            for j in range(0, K):
                for k in range(1, len(solution[j])):
                    costinc = data[solution[j][k - 1], i] + data[i, solution[j][k]] \
                              - data[solution[j][k - 1], solution[j][k]]
                    if costinc < bestcostinc:
                        bestcostinc = costinc
                        bestcus = i
                        bestveh = j
                        bestpos = k
        cuslist.remove(bestcus)
        solution[bestveh].insert(bestpos,bestcus)
        cost[bestveh] += bestcostinc

# main loop for simulated annealing
    temperature = 0.1
    bestsol = solution
    bestcost = max(cost)
    tempsol = solution

    cuslist = []
    for i in range(K, N):
        cuslist.append(i)

    for i in range(0,5000):
        solution = tempsol
        relocate(solution,cost,cuslist,data)
        if (max(cost)-bestcost)/bestcost < temperature:
            tempsol = solution
            if max(cost) < bestcost:
               bestsol = solution
               bestcost = max(cost)
        temperature -= temperature/5000

    end = time.time()

    print("maxcost:", max(cost))
    print("totalcost", sum(cost))
    print(solution)
    print("the route is:", solution[cost.index(max(cost))])
    print("computational time:", end - start)

if __name__=="__main__":
    cost_matrix = np.loadtxt('outfile.txt')
    simulated_annealing(10, 20, cost_matrix)
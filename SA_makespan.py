import numpy as np
import time


# this function relocate the position of each customer and find the move with lowest cost increment
def relocate(solution,cost,cuslist,data):
    bestcost = 10000
    for i in cuslist:
        tempcost = np.ones(len(cost)) * cost
        for j in range(0,len(solution)):
            if i in solution[j]:
               pos = solution[j].index(i)
               costinc = - data[solution[j][pos-1],i] - data[i,solution[j][pos+1]] \
                          + data[solution[j][pos-1],solution[j][pos+1]]
               tempcost[j] += costinc
               changedroute = j
        for j in range(0, len(solution)):
            if j != changedroute:
                for k in range(1, len(solution[j])):
                    newtempcost = np.ones(len(tempcost)) * tempcost
                    tempcostinc = costinc
                    tempcostinc += data[solution[j][k - 1], i] + data[i, solution[j][k]] \
                                   - data[solution[j][k - 1], solution[j][k]]
                    costincrease = data[solution[j][k - 1], i] + data[i, solution[j][k]] \
                                   - data[solution[j][k - 1], solution[j][k]]
                    newtempcost[j] += costincrease
                    # if tempcostinc < bestcost:
                    if max(newtempcost) < bestcost:
                        bestcostinc = costincrease
                        bestcost = max(newtempcost)
                        bestveh = j
                        bestpos = k
                        bestcus = i
            else:
                tempsolution = solution[j].copy()
                tempsolution.remove(i)
                for k in range(1, len(tempsolution)):
                    newtempcost = np.ones(len(tempcost)) * tempcost
                    tempcostinc = costinc
                    tempcostinc += data[tempsolution[k - 1], i] + data[i, tempsolution[k]] \
                                   - data[tempsolution[k - 1], tempsolution[k]]
                    costincrease = data[tempsolution[k - 1], i] + data[i, tempsolution[k]] \
                                   - data[tempsolution[k - 1], tempsolution[k]]
                    newtempcost[j] += costincrease
                    # if tempcostinc < bestcost:
                    if max(newtempcost) < bestcost:
                        bestcostinc = costincrease
                        bestcost = max(newtempcost)
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
        bestcost = 10000
        for i in cuslist:
            for j in range(0, K):
                for k in range(1, len(solution[j])):
                # optimize total length
                   costinc = data[solution[j][k - 1], i] + data[i, solution[j][k]] \
                             - data[solution[j][k - 1], solution[j][k]]
                   # if costinc < bestcostinc:
                # optimize max distance
                   tempcost = np.ones(len(cost))*cost
                   tempcost[j] += data[solution[j][k - 1], i] + data[i, solution[j][k]] \
                           - data[solution[j][k - 1], solution[j][k]]
                   if max(tempcost) < bestcost:
                      bestcost = max(tempcost)
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

    for i in range(0,500):
        solution = tempsol
        relocate(solution,cost,cuslist,data)
        if (max(cost)-bestcost)/bestcost < temperature:
            tempsol = solution
            if max(cost) < bestcost:
               bestsol = solution
               bestcost = max(cost)
        temperature -= temperature/5000

    end = time.time()

    # print("maxcost:", max(cost))
    # print(cost)
    # print(solution)
    # print("total:",sum(cost))
    # print("the route is:", solution[cost.index(max(cost))])
    # print("computational time:", end - start)
    # solution indexed from targets
    solution = [[(x - K) for x in row] for row in solution]
    assignment_result = [i[1:-1] for i in solution]
    return assignment_result, max(cost), sum(cost), end-start

    # # double check
    # length = 0
    # id = cost.index(max(cost))
    # for i in range(0,len(solution[id])-1):
    #     length += data[solution[id][i]][solution[id][i+1]]
    #
    # print(length)
    #
    # total = 0
    # for i in range(0, K):
    #     length = 0
    #     for j in range(0, len(solution[i]) - 1):
    #         length += data[solution[i][j]][solution[i][j + 1]]
    #         print(data[solution[i][j]][solution[i][j + 1]])
    #     total += length
    #     print(solution[i],length)
    # print(total)



if __name__=="__main__":
    cost_matrix = np.loadtxt('outfile.txt')
    assignment_result, max_cost, total_cost, time_SA = simulated_annealing(10, 20, cost_matrix)
import numpy as np
import time

filename = 'outfile.txt'

def main(filename):

    start = time.time()

# load file
    data = np.loadtxt(filename)

# k is the number of vehicles, c is the number of customers
    K = 10
    C = 20
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

    end = time.time()
    print(solution)
    print("maxcost:", max(cost))
    print("the route is:", solution[cost.index(max(cost))])
    print("computational time:", end - start)

main(filename)
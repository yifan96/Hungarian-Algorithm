from sklearn.cluster import KMeans
import numpy as np
import pyomo.environ as pyEnv
X = np.array([[-1400,20], [0,0], [400,-1000],[1000, 1200], [-1000, -500], [-100, 600], [-900, 1000]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print(kmeans.labels_)





def TSP(cost_matrix):
    n = len(cost_matrix)
    # Model
    model = pyEnv.ConcreteModel()

    # Indexes for the cities
    model.M = pyEnv.RangeSet(n)
    model.N = pyEnv.RangeSet(n)

    # Index for the dummy variable u
    model.U = pyEnv.RangeSet(2, n)
    # Decision variables xij
    model.x = pyEnv.Var(model.N, model.M, within=pyEnv.Binary)

    # Dummy variable ui
    model.u = pyEnv.Var(model.N, within=pyEnv.NonNegativeIntegers, bounds=(0, n - 1))
    # Cost Matrix cij
    model.c = pyEnv.Param(model.N, model.M, initialize=lambda model, i, j: cost_matrix[i - 1][j - 1])

    def obj_func(model):
        return sum(model.x[i, j] * model.c[i, j] for i in model.N for j in model.M)

    model.objective = pyEnv.Objective(rule=obj_func, sense=pyEnv.minimize)

    def rule_const1(model, M):
        return sum(model.x[i, M] for i in model.N if i != M) == 1

    model.const1 = pyEnv.Constraint(model.M, rule=rule_const1)

    def rule_const2(model, N):
        return sum(model.x[N, j] for j in model.M if j != N) == 1

    model.rest2 = pyEnv.Constraint(model.N, rule=rule_const2)

    def rule_const3(model, i, j):
        if i != j:
            return model.u[i] - model.u[j] + model.x[i, j] * n <= n - 1
        else:
            # Yeah, this else doesn't say anything
            return model.u[i] - model.u[i] == 0

    model.rest3 = pyEnv.Constraint(model.U, model.N, rule=rule_const3)
    # Prints the entire model
    # model.pprint()
    # Solves
    solver = pyEnv.SolverFactory('cplex_direct')
    result = solver.solve(model, tee=False)


    l = list(model.x.keys())

    aa = []
    for i in l:
        if model.x[i]() != 0:
            #print(i, '--', model.x[i]())
            aa.append(i)
    j = 0
    order = [1]
    while j != 1:
        if j != 0:
            j = aa[j - 1][1]
        else:
            j = aa[j][1]
        order.append(j)

    print(order)









# quadrotor 1: -1500 -2
# quadrotor 2: -1510 -2


# task assigned to quadrotor 1:
tasks_1 = np.where(np.array(kmeans.labels_)>0)[0]
tasks_2 = np.where(np.array(kmeans.labels_)==0)[0]
#print(tasks_1)

X_1 = np.take(X, tasks_1, axis=0)
X_1 = np.insert(X_1, 0, [-1500,-2], axis=0)
#print(X_1)
cost_matrix_1 = []
n = len(X_1)
print(n)
cost_matrix_1 = [[0 for i in range(n)] for i in range(n)]
for i in range(n):
    for j in range(n):
        if j > i:
            cost_matrix_1[i][j] = np.linalg.norm(X_1[i]-X_1[j])
        if j == i:
            cost_matrix_1[i][j] = 9999
        if j < i:
            cost_matrix_1[i][j] = cost_matrix_1[j][i]
for i in range(1,n):
    cost_matrix_1[i][0] = 0
print('cost_matrix_1')
print(cost_matrix_1)
TSP(cost_matrix_1)


# task assigned to quadrotor 2
X_2 = np.take(X, tasks_2, axis=0)
X_2 = np.insert(X_1, 0, [-1500,4], axis=0)
#print(X_1)
cost_matrix_2 = []
n = len(X_2)

cost_matrix_2 = [[0 for i in range(n)] for i in range(n)]
for i in range(n):
    for j in range(n):
        if j > i:
            cost_matrix_2[i][j] = np.linalg.norm(X_2[i]-X_2[j])
        if j == i:
            cost_matrix_2[i][j] = 9999
        if j < i:
            cost_matrix_2[i][j] = cost_matrix_2[j][i]
for i in range(1,n):
    cost_matrix_2[i][0] = 0
print('cost_matrix_2')
print(cost_matrix_2)
TSP(cost_matrix_2)
import pyomo.environ as pyEnv
import rospy
file = open('distance_array.txt')
lines = file.readlines()
file.close()
cost_matrix = []
# for i in range(len(lines)):
#     aux = lines[i][:-1].split()
#     aux = [float(i) for i in aux if i!= '']
#     cost_matrix.append(aux)
# print(cost_matrix)

#group 1
# cost_matrix = [[9999, 48.5768, 28.1421, 43.8549, 44.7846],
#                [0, 9999, 27.46, 64.83, 79.71],
#                [0, 27.46, 9999, 40.29, 53.51],
#                [0, 64.83, 40.29, 9999, 18.59],
#                [0, 79.71, 53.51, 18.59, 9999]]

# group 2
# cost_matrix = [[9999, 34.4853, 38.2843, 58.0492],
#                [0, 9999, 36.97, 24.14],
#                [0, 36.97, 9999, 40.63],
#                [0, 24.14, 40.63, 9999]]

#group 3
# cost_matrix = [[9999, 49.9479, 49.1210, 39.9494],
#                [0, 9999, 33.41, 30.29],
#                [0, 33.41, 9999, 10.83],
#                [0, 30.29, 10.83, 9999]]


# lab experiment  start: a   ; goal: 1,2,3,4,6
# cost_matrix = [[9999, 2919.83, 2911.63, 2938.60, 2950.80, 2930.22],
#                [0, 9999, 13.66, 35.66, 42.98, 18.58],
#                [0, 13.66, 9999, 31.76, 40.43, 23.07],
#                [0, 35.66, 31.76, 9999, 19.32, 33.86],
#                [0, 42.98, 40.43, 19.32, 9999, 33.18],
#                [0, 18.58, 23.07, 33.86, 33.18, 9999]]
# lab experiment  start: b   ; goal: 5,7
# cost_matrix = [[9999, 78.5344, 71.3546, 49.0703],
#                [0, 9999, 30.9774, 31.8988],
#                [0, 30.9774, 9999, 41.4135],
#                [0, 31.8988, 41.4135, 9999]]



# lab experiment four cases:
# cost_matrix = [[9999, 78.5344, 71.3546, 49.0703],
#                [0, 9999, 30.9774, 31.8988],
#                [0, 30.9774, 9999, 41.4135],
#                [0, 31.8988, 41.4135, 9999]]
# cost_matrix = [[9999, 42.31, 46.83],
#                [0, 9999, 18.24],
#                [0, 18.24, 9999]]

# cost_matrix = [[9999, 4.83, 17.41, 19.95],
#                [0, 9999, 13.76, 17.76],
#                [0, 13.76, 9999, 5.66],
#                [0, 17.76, 5.66, 9999]]
cost_matrix =[[9999999, 102, 705, 1167],
              [0, 9999999, 656, 1100],
              [0, 705, 9999999, 1503],
              [0, 1167, 1100, 99999999]]
n = len(cost_matrix)
#Model
model = pyEnv.ConcreteModel()

#Indexes for the cities
model.M = pyEnv.RangeSet(n)
model.N = pyEnv.RangeSet(n)

#Index for the dummy variable u
model.U = pyEnv.RangeSet(2,n)
#Decision variables xij
model.x = pyEnv.Var(model.N,model.M, within=pyEnv.Binary)

#Dummy variable ui
model.u = pyEnv.Var(model.N, within=pyEnv.NonNegativeIntegers,bounds=(0,n-1))
#Cost Matrix cij
model.c = pyEnv.Param(model.N, model.M,initialize=lambda model, i, j: cost_matrix[i-1][j-1])
def obj_func(model):
    return sum(model.x[i,j] * model.c[i,j] for i in model.N for j in model.M)

model.objective = pyEnv.Objective(rule=obj_func,sense=pyEnv.minimize)
def rule_const1(model,M):
    return sum(model.x[i,M] for i in model.N if i!=M ) == 1

model.const1 = pyEnv.Constraint(model.M,rule=rule_const1)
def rule_const2(model,N):
    return sum(model.x[N,j] for j in model.M if j!=N) == 1

model.rest2 = pyEnv.Constraint(model.N,rule=rule_const2)


def rule_const3(model, i, j):
    if i != j:
        return model.u[i] - model.u[j] + model.x[i, j] * n <= n - 1
    else:
        # Yeah, this else doesn't say anything
        return model.u[i] - model.u[i] == 0


model.rest3 = pyEnv.Constraint(model.U, model.N, rule=rule_const3)
#Prints the entire model
#model.pprint()
#Solves
solver = pyEnv.SolverFactory('cplex_direct')
result = solver.solve(model,tee = False)

#Prints the results
print(result)

l = list(model.x.keys())
print(l)
aa=[]
for i in l:
    if model.x[i]() != 0:
        print(i,'--', model.x[i]())
        aa.append(i)
j=0
order=[1]
while j != 1:
    if j !=0:
        j=aa[j-1][1]
    else:
        j=aa[j][1]
    order.append(j)
    print(j)
print(order)
# print(aa[2][1])

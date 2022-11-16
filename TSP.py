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

cost_matrix = [[9999, 78.5344, 71.3546, 49.0703],
               [0, 9999, 30.9774, 31.8988],
               [0, 30.9774, 9999, 41.4135],
               [0, 31.8988, 41.4135, 9999]]

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


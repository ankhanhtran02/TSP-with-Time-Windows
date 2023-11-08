from ortools.sat.python import cp_model

class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    # print intermediate solution
    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
    def on_solution_callback(self):
        self.__solution_count +=1
        for v in self.__variables:
            print('%s =  %i |'% (v,self.Value(v)), end = ' ')
        print()
    def solution_count(self):
        return self.__solution_count

def inp():
    num_var = int(input()) # not including starting point & fake ending point
    start = []
    finish = []
    service_time = []
    for _ in range(num_var):
        a,b,c = map(int,input().split())
        start.append(a)
        finish.append(b)
        service_time.append(c)
    service_time = [0]+service_time+[0] # add d[0] = 0, d[N+1] = 0
    start = [0]+start+[0]
    finish = [0]+finish+[0]
    delivery_time = []
    for _ in range(num_var+1):
        delivery_time.append(list(map(int,input().split()))+[0])
    delivery_time.extend([[0]*(num_var+2)])
    return num_var, start, finish, service_time, delivery_time

num_var, start, finish, service_time, delivery_time = inp()

model = cp_model.CpModel()

W=0 # W: upper bound for total time
for i in range(num_var+2):
    for j in range(num_var+2):
        W += delivery_time[i][j]

x = [[model.NewIntVar(0, 1, f"x[{i},{j}]") for i in range(0,num_var+2)] for j in range(0,num_var+2) ] # i: source, j: destination
y = [model.NewIntVar(0, 9999999, f"y[{i}]") for i in range(0,num_var+2) ]

# D_source 0-> N
# D_destination  1 -> N+1

# Contraints:
model.Add(y[0]==0)

for i in range(num_var+2):
    model.Add(x[i][i]==0)               # source and destination are not the same
    sum_des = 0             
    sum_source = 0
    for j in range(num_var+2):
        sum_des += x[i][j]
        sum_source += x[j][i]
    model.Add(sum_des==1)               # only leave i once
    model.Add(sum_source==1)            # only go to i once
    model.Add(x[num_var+1][i]==0)       # must end at N+1
    model.Add(x[i][0]==0)               # must start at 0

for i in range(1,num_var+1):            # for every customer i from 1 to N
    model.Add(y[i] >= start[i])         # arrive after start(i)
    model.Add(y[i] <= finish[i])        # arrive before finish(i)
    b=model.NewBoolVar('b')  
    for j in range(1,num_var+1): 
        # if go from i->j, add service & delivery time at i to total time
        model.Add(x[i][j] == 1).OnlyEnforceIf(b)
        model.Add(x[i][j] != 1).OnlyEnforceIf(b.Not())
        model.Add(y[j] == y[i] + service_time[i] + delivery_time[i][j]).OnlyEnforceIf(b) 

model.Minimize(y[num_var+1])

solver = cp_model.CpSolver()
status = solver.Solve(model)
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(f'Minimum objective function: {solver.ObjectiveValue()}')
    temp = 0
    for i in range(num_var+2):
        for j in range(num_var+2):
            if i==temp and x[i][j]==1:
                print(f'{i} -> {j}: y[{j}] = {y[j]}')
                temp = j
else:
    print('No solution found.')



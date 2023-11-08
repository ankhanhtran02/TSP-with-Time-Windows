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

B = [int(i) for i in range(num_var+2)]
B2 = [(i,j) for i in B for j in B]
F1 = [(int(i),0) for i in B]
F2 = [(int(num_var+1),int(i)) for i in B]
F3 = [(i,i) for i in B]
A = list(filter(lambda item: item not in F1 and item not in F2 and item not in F3, B2))

W=0 # W: upper bound for total time
for i in range(num_var+2):
    W+=service_time[i]
    for j in range(num_var+2):
        W += delivery_time[i][j]

x = [[model.NewIntVar(0, 1, f"x[{i},{j}]") for i in range(0,num_var+2)] for j in range(0,num_var+2) ] # i: source, j: destination
y = [model.NewIntVar(0, W, f"y[{i}]") for i in range(0,num_var+2) ]

# D_source 0-> N
# D_destination  1 -> N+1

# Contraints:
model.Add(y[0]==0)

# only leave i once
# A_des(i) = {j | (i,j) in A} = [edge[1] for edge in A if edge[0]==i]
for i in range(1, num_var+1):
    model.Add(sum([x[i][j] for j in [edge[1] for edge in A if edge[0]==i]]) == 1)

# only go to i once
# A_start(i) = {j | (j,i) in A} = [edge[0] for edge in A if edge[1]==i]
for i in range(1, num_var+1):
    model.Add(sum([x[j][i] for j in [edge[0] for edge in A if edge[1]==i]]) == 1)

model.Add(sum([x[0][j] for j in range(1,num_var+1)])==1)
model.Add(sum([x[j][num_var+1] for j in range(1,num_var+1)])==1)

for i,j in A:           
    b=model.NewBoolVar('b')  
        # if go from i->j, add service & delivery time at i to total time
    model.Add(x[i][j] == 1).OnlyEnforceIf(b)
    model.Add(x[i][j] != 1).OnlyEnforceIf(b.Not())
    model.Add(y[j] == y[i] + service_time[i] + delivery_time[i][j]).OnlyEnforceIf(b)

for i in range(1,num_var+1):
    model.Add(y[i] >= start[i])
    model.Add(y[i] <= finish[i])

model.Minimize(y[num_var+1])

solver = cp_model.CpSolver()
status = solver.Solve(model)
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(f'Minimum objective function: {solver.ObjectiveValue()}')
    for i in range(num_var+2):
        for j in range(num_var+2):
            if solver.Value(x[i][j]) ==1:
                print(f'{i} -> {j}')
else:
    print('No solution found.')



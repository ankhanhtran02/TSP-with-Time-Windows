from ortools.sat.python import cp_model
import pandas as pd

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

def read_input_file(filename='TSPTW_test_1.txt'):
    '''
    This function reads an input file for the TSPTW problem.
    If successful, this returns a tuple (N, e, l, d, t) where:
        + N is the number of points to visit
        + e and l contain the starts and ends of N time windows
        + d is the service times at N points
        + t is the travel time matrix, size (N+1) x (N+1)
    '''
    try:
        with open(filename, 'r') as file_handle:
            content = file_handle.read().split('\n')
            N = int(content[0])
            e, l, d = [0 for _ in range(N)], [0 for _ in range(N)], [0 for _ in range(N)]
            t = []
            for i in range(N):
                e[i], l[i], d[i] = map(int, content[i+1].split())
            d = [0]+d+[0] # add d[0] = 0, d[N+1] = 0
            e = [0]+e+[0]
            l = [0]+l+[999999]
            for i in range(N+1):
                t.append(list(map(int, content[i+N+1].split()))+[0])
            t.extend([[0]*(N+2)])
            return (N, e, l, d, t)
    except FileNotFoundError:
        return None
    except:
        print('Unknown error when reading file!')
        return None

def inp():
    N = int(input()) # number of customers, not including starting point 
    e = []
    l = []
    d = []
    for _ in range(N):
        a,b,c = map(int,input().split())
        e.append(a)
        l.append(b)
        d.append(c)
    d = [0]+d+[0] # add d[0] = 0, d[N+1] = 0
    e = [0]+e+[0]
    l = [0]+l+[999999999]
    t = []
    for _ in range(N+1):
        t.append(list(map(int,input().split()))+[0])
    t.extend([[0]*(N+2)])
    return N, e, l, d, t

def solve_tsptw(N, e, l, d, t):
  model = cp_model.CpModel()

  # B = [int(i) for i in range(N+2)]
  # B2 = [(i,j) for i in B for j in B]
  # F1 = [(int(i),0) for i in B]
  # F2 = [(int(N+1),int(i)) for i in B]
  # F3 = [(i,i) for i in B]
  # A = list(filter(lambda item: item not in F1 and item not in F2 and item not in F3, B2))

  A = []
  for i in range(N+2):
      for j in range(N+2):
          if j!= 0 and i!=(N+1) and i!=j:
              A.append((i, j))

  x = [[model.NewIntVar(0, 1, f"x[{i},{j}]") for i in range(0,N+2)] for j in range(0,N+2) ] # i: source, j: destination
  y = [model.NewIntVar(0, 999999, f"y[{i}]") for i in range(0,N+2) ] # arrival time

  # Contraints:
  model.Add(y[0]==0)

  # only leave i once
  # A_des(i) = {j | (i,j) in A} = [edge[1] for edge in A if edge[0]==i]
  for i in range(1, N+1):
      model.Add(sum([x[i][j] for j in [edge[1] for edge in A if edge[0]==i]]) == 1)

  # only go to i once
  # A_source(i) = {j | (j,i) in A} = [edge[0] for edge in A if edge[1]==i]
  for i in range(1, N+1):
      model.Add(sum([x[j][i] for j in [edge[0] for edge in A if edge[1]==i]]) == 1)

  model.Add(sum([x[0][j] for j in range(1,N+1)])==1)
  model.Add(sum([x[j][N+1] for j in range(1,N+1)])==1)

  for i,j in A:
      b=model.NewBoolVar('b')
      max_value = model.NewIntVar(0, 999999 , 'max_value')
      model.AddMaxEquality(max_value, [y[i], e[i]])
      model.Add(x[i][j] == 1).OnlyEnforceIf(b)
      model.Add(x[i][j] != 1).OnlyEnforceIf(b.Not())
      model.Add(y[j] == max_value + d[i] + t[i][j]).OnlyEnforceIf(b)

  for i in range(1,N+1):
      model.Add(y[i] <= l[i])

  model.Minimize(sum([x[i][j]*t[i][j] for i in range(N+2) for j in range(N+2)]))

  solver = cp_model.CpSolver()

  solver.parameters.max_time_in_seconds = 180
  status = solver.Solve(model)
  runtime = solver.WallTime()
  if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
      l = []
      i=0
      while len(l) < N:
          for j in range(N+1):
              if solver.Value(x[i][j])==1:
                  l.append(j)
                  i=j
                  break
      return l, solver.ObjectiveValue(), runtime 
  else:
      return 'None', 'None', 'None'

N, e, l, d, t = inp()
path, value, runtime = solve_tsptw(N, e, l, d, t)
print(N)
print(*path)

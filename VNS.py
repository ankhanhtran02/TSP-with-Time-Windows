import numpy as np
import random
from timeit import default_timer

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
    d = [0]+d
    e = [0]+e
    l = [0]+l
    t = []
    for _ in range(N+1):
        t.append(list(map(int,input().split())))
    return N, e, l, d, t


def CheckFeasible(x: list): # x: a solution
    if x == None:
        return False
    # y={0:0,} # y: arrival time
    # s = 0
    # for i in x:
    #     y_i = max(y[s],e[s])+d[s]+t[s][i]
    #     if y_i>l[i]:
    #         return False
    #     else:
    #         s=i
    #         y[i] = y_i
    # return True
    return ObjFunc(x) == 0

def CheckViolated(a, x): # a: node in x
    y={0:0,}
    s=0
    for i in x:
        y_i = max(y[s],e[s])+d[s]+t[s][i]
        if i==a:
            y[i]=y_i
            break
        else:
            s=i
            y[i]=y_i
    if y[i] > l[i]:
        return True
    return False

def ObjFunc(x: list):
    y={0:0,} # y: arrival time
    i = 0
    s=0
    for j in x:
        y[j] = max(y[i],e[i])+d[i]+t[i][j]
        s+=max(0,y[j]-l[j])
        i=j
    return s
        

# def Local1Shift(x: list):
#     neighbors = []
#     violated_nodes = list(filter(lambda i: CheckViolated(i,x),x))
#     if len(violated_nodes)==0:
#         return x
#     for i in violated_nodes:
#         positions = list(range(0,x.index(i)))
#         for p in positions:
#             temp = x.copy()
#             temp.remove(i)
#             temp.insert(p,i)
#             if temp not in neighbors:
#                 neighbors.append(temp)
#     current_objective = ObjFunc(x)
#     neighbors_objective = list(map(ObjFunc,neighbors))
#     min_obj = min(min(neighbors_objective),current_objective)
#     all_sol = neighbors + [x]
#     min_x_list = list(filter(lambda sol: ObjFunc(sol) == min_obj, all_sol))
#     new_x = min_x_list[0]
#     return new_x

def Local1Shift(x: list):
    current_obj = ObjFunc(x)
    violated_nodes = list(filter(lambda i: CheckViolated(i,x),x))
    unviolated_nodes = list(filter(lambda i: i not in violated_nodes,x))
    for i in violated_nodes:
        for position in list(range(0,x.index(i))): # backward movements of violated customers
            neighbor = x.copy()
            neighbor.remove(i)
            neighbor.insert(position,i)
            if ObjFunc(neighbor) < current_obj:
                return neighbor
    for i in unviolated_nodes:
        for position in list(range(x.index(i)+1,len(x)+1)): # forward movements of non-violated customers
            neighbor = x.copy()
            neighbor.remove(i)
            neighbor.insert(position,i)
            if ObjFunc(neighbor) < current_obj:
                return neighbor
    for i in unviolated_nodes:
        for position in list(range(0,x.index(i))): # backward movements of non-violated customers
            neighbor = x.copy()
            neighbor.remove(i)
            neighbor.insert(position,i)
            if ObjFunc(neighbor) < current_obj:
                return neighbor
    for i in violated_nodes:
        for position in list(range(x.index(i)+1,len(x)+1)): # forward movements of violated customers
            neighbor = x.copy()
            neighbor.remove(i)
            neighbor.insert(position,i)
            if ObjFunc(neighbor) < current_obj:
                return neighbor
    return x

def Pertubation(x,level):
    new_sol = x.copy()
    for _ in range(level):
        i,j = random.sample(range(len(new_sol)),2)
        new_sol[i], new_sol[j] = new_sol[j], new_sol[i]
    return new_sol

def solve(level_max):
    x = None
    start_timer = default_timer()
    while not CheckFeasible(x):
        print('---------------')
        level = 1
        x = random.sample(list(range(1,N+1)),N) # random initial solution
        # x = sorted(list(range(1,N+1)), key = lambda i: l[i]) # sorted initial solution
        print(f'x = {x},',end=' ')
        x = Local1Shift(x)
        print(f'after local search = {x}, feasible = {CheckFeasible(x)}')
        while (not CheckFeasible(x)) and (level <= level_max):
            print(f'level = {level}:',end = ' ')
            x1 = Pertubation(x,level)
            print(f'x1 = {x1}',end=' ')
            x1 = Local1Shift(x1)
            print(f'after local search = {x1}')
            print(f'Obj value: x1 = {ObjFunc(x1)}, x = {ObjFunc(x)}')
            if ObjFunc(x1) <= ObjFunc(x):
                x=x1
                level=1
            else:
                level+=1
    end_timer = default_timer()
    print(f'Execution time is: {end_timer - start_timer}')
    return x

N, e, l, d, t = inp()

print(N)
print(' '.join(list(map(str,solve(8)))))

    








import time
Max_calculation = 30000000

n = int(input())
e = [-1]
l = [-1]
d = [-1]
for i in range(n):
    _ = list(map(int, input().split()))
    e.append(_[0])
    l.append(_[1])
    d.append(_[2])

t = []
for i in range(n + 1):
    _ = list(map(int, input().split()))
    t.append(_)

greedy = sorted(l)
start_path = []
for i in range (1, n + 1):
    for j in range (1, n + 1):
        if greedy[i] == l[j]:
            start_path.append(j)
            break

tabu = []
tabu.append(start_path)

def Calculate(path):
    check = True
    cur_pos = 0
    total_time = 0
    travel_time = 0
    for i in range (0, n):
        total_time += t[cur_pos][path[i]]
        travel_time += t[cur_pos][path[i]]
        if total_time <= l[path[i]]:   
            total_time = max(total_time, e[path[i]])
            total_time += d[path[i]]
            cur_pos = path[i]
        else: 
            check = False
            break
    travel_time += t[path[n - 1]][0]
    # print(travel_time)
    if check == True:
        return travel_time
    else:   
        return 99999999999

iteration = 0
while iteration < Max_calculation // (n * n): 
    element = tabu.pop() 
    # print(element)
    cur_best_element = element.copy()
    check = Calculate(cur_best_element)
    # print(check)
    Min = 99999999999
    for i in range (0, n):
        for j in range (0, n):
            try_test = element.copy()
            if i != j:
                try_test[i], try_test[j] = try_test[j], try_test[i]
                # print(try_test, end = ' ')
                # print(Calculate(try_test))
                if Calculate(try_test) < Min:
                    cur_best_element = try_test
                    Min = Calculate(try_test)
    # print("Interation: ", iteration, end = ' ')
    # print(tabu)
    # print(Min)
    # print(check)
    if Min >= check:
        tabu.append(element)
        break 
    else: 
        tabu.append(cur_best_element)
        iteration += 1
print(n)
for i in tabu[0]:
    print(i, end = ' ')
            
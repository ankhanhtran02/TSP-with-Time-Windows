import time

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

Min = 99999999999
check = [False for i in range (n + 1)]
path = [0 for i in range (n + 1)]
ans_path = None

def Calculate(path):
    global Min, ans_path
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
    print(path)
    if travel_time < Min and check == True:
        ans_path = path
        Min = travel_time
        print("FOUND A SOLUTION: ", path)

def Try(k):
    for i in range (1, n + 1):
        if check[i] == False:
            path[k] = i
            check[i] = True
            if k == n:
                Calculate(path[1:])
            else:
                Try(k + 1)
            check[i] = False

Try(1)
print(n)
for i in range (0, n):
    print(ans_path[i], end = ' ')
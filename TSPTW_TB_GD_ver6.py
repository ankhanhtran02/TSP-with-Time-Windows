from collections import deque
import time
start_time = time.time()

'''
    Khai báo, đọc từ file ra. Lưu lần lượt vào các list e, l, d, t
'''
with open('testcase/B1000.txt', 'r') as file:
    n = int(file.readline().strip())
    e = [-1]
    l = [-1]
    d = [-1]
    for _ in range(n):
        e_, l_, d_ = map(int, file.readline().strip().split())
        e.append(e_)
        l.append(l_)
        d.append(d_)
    t = []
    for _ in range(n + 1):
        t.append(list(map(int, file.readline().strip().split())))

'''
    Sắp xếp e, l vào 1 tuple sau đó sắp xếp lại tuple theo e (e bằng nhau thì theo l)
    Đây sẽ là start_path - đường đi thỏa mãn các ràng buộc đề bài, cho đây là đường đi xuất phát và cần tối ưu
'''

def start_path(a):
    greedy = list(a)
    greedy.sort()

    start_path_visited = [False for i in range (n + 1)]
    path = []
    for i in range (1, n + 1):
        for j in range (1, n + 1):
            if greedy[i] == a[j] and start_path_visited[j] == False:
                path.append(j)
                start_path_visited[j] = True
                break
    return path

# choose the start path as sored by (e, l) and (l, e)
start_path1 = start_path(e)
start_path2 = start_path(l)
tabu = deque()
tabu.append((1, start_path1))
tabu.append((1, start_path2))

'''
    Sử dụng thuật toán tabu để tìm đường đi tối ưu hơn đường đi trước:
    - Hill climbing: Idea là đổi chỗ vị trí của 2 địa điểm cho nhau, nếu thấy nó tối ưu hơn đường đi trước thì sẽ tiếp tục đổi chỗ từ đó
    - Tabu Search: Trong trường hợp nếu đổi chỗ 2 địa điểm mà đường đi không tối ưu hơn, thì chấp nhận sẽ giữ đường đi đó và hoán đổi thêm k lần nữa (chấp nhận chịu thiệt để xem xem đổi chỗ 2 điểm ko tối ưu thì 3, 4, ... điểm có tối ưu hơn ko)
    - Code:
        + Tabu: một list (deque) trong đó lưu 2 giá trị
            • Giá trị đầu tiên lưu số k - số lần chấp nhận đi đường đi thiệt thòi để tìm local maximum tốt hơn
            • Giá trị thứ 2 lưu đường đi hiện tại 
        + Calculate: tính toán giá trị của đường đi 
        + list_optimal_path: lưu lại những đường đi thỏa mãn ràng buộc đề bài và tốt hơn giá trị start_path
''' 

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
    if check == True:
        return travel_time
    else:   
        return 99999999999

value_path1 = Calculate(start_path1)
value_path2 = Calculate(start_path2)
cur_MIN = min(value_path1, value_path2)
prev = 99999999999
optimal_path = None
if cur_MIN == value_path1:
    optimal_path = start_path1
else: optimal_path = start_path2

'''
    Trong vòng lặp while True, chạy đến khi nào thời gian tối đa cho phép (tự đặt)
    Trong đó:
        - nếu phần tử được pop từ list tabu thỏa mãn ràng buộc đề bài, lưu vào list_optimal_path
        - trong trường hợp phần tử được pop ra đang là đường đi chấp nhận chịu thiệt, có 2 khả năng:
            + nếu đường đi nhanh hơn cur_MIN hiện tại ==> đã tìm được một local_maximum mới ==> found_better = True khi đó sẽ xóa toàn bộ phần tử trong tabu và tiếp tục tìm kiếm từ đường đi đấy, reset giá trị đầu tiên về 0 và lưu vào cur_best_element
            + nếu đường đi nhanh = cur_MIN hiện tại ==> thêm nó vào cur_best_element
        - sau khi xử lý xong, thêm các phần tử trong cur_best_element vào deque tabu và tiếp tục vòng while
    Các tham số có thể thay đổi nhằm tối ưu:
        + time.time() - start_time > x : cho phép thời gian tối đa = x
        + element[0] <= x : cho phép đường đi chịu thiệt x lần, tức là cho phép đổi chỗ 2 địa điểm x lần nhằm tìm ra đường đi tối ưu
        + for j in range (i + 1, min(i + k, n)) : duyệt nhiều nhất k phần tử sau i, nó sẽ giúp mình xét đường đi nếu đổi chỗ địa điểm i-th với địa điểm i + j-th (j thuộc khoảng (i, i + k)) 
'''

# change the parameter for better understanding
run_time = 180
depth = 2
vertex_check = 2

while len(tabu) > 0: 
    element = tabu.popleft()
    if element[0] <= depth:
        cur_best_element = [] 
        found_better = False
        for i in range (0, n - 1):
            for j in range (i + 1, min(i + vertex_check, n)):
                try_path = element[1][:]
                try_path = try_path[:i] + try_path[i + 1:j] + try_path[i:i+1] + try_path[j:]
                travel_time = Calculate(try_path)
                if travel_time < cur_MIN:
                    found_better = True 
                    cur_best_element = []
                    cur_best_element.append((0, try_path[:]))
                    optimal_path = try_path[:]
                    cur_MIN = travel_time 
                elif travel_time == cur_MIN:
                    cur_best_element.append((0, try_path))
                else:
                    cur_best_element.append((element[0] + 1, try_path))
                # else: cur_best_element.append((element[0] + 2, try_path))
                if time.time() - start_time > run_time: 
                    break
        # print("TABU LENGTH", len(tabu), "CUR_MIN:", cur_MIN)
        if found_better == True and len(cur_best_element) > 0:
            print("TABU LENGTH", len(tabu), "CUR_MIN:", cur_MIN, "DEPTH:", element[0])
            tabu.clear()
        for i in cur_best_element:
            tabu.append(i)
    if time.time() - start_time > run_time: 
        break

print("TIME_TAKEN:", time.time() - start_time)
print("START_PATH:", start_path1)
print("START_PATH_VALUE:", Calculate(start_path1))
print("CURRENT_OPTIMAL_PATH:", optimal_path)
print("CURRENT_OPTIMAL_PATH_VALUE:", Calculate(optimal_path))

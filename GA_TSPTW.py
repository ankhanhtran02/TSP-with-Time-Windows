'''
Step 1: Initialization

Step 2: Selection
    
Step 3: Crossover
Step 4: Mutation
    
Step 5: Replacement
    Create a new population by combining offspring chromosomes with selected parent chromosomes.
    Elitism ensures that the best chromosomes are preserved across generations.
    
Step 6: Termination
    Repeat steps 2-5 until a stopping criterion is met (e.g., maximum generations reached, no improvement in solution).
    The best chromosome in the final population represents the optimal or near-optimal route for the TSPTW problem.
'''

# from cProfile import run
import random
# random.seed(12345)
import time
### mn thay tên file là ra đáp án nhé, còn điều chỉnh tham số thì ở dòng tsptw_ga ( 268)

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
            for i in range(N+1):
                t.append(list(map(int, content[i+N+1].split())))
            return (N, e, l, d, t)
    except FileNotFoundError:
        return None
    except:
        print('Unknown error when reading file!')
        return None
    
    
# n , e, l, d, time_matrix = read_input_file('r1.txt')
# print(n,e,l,d)
# print([n,e,l,d,time_matrix])
# n , e, l, d, time_matrix = read_input_file('Test_BinhQuan.txt')
'''
hoặc nhập input thủ công như đống # dưới đây
'''

n = int(input())
e = [0]
l = [0]
d = [0]

for i in range(n):
    _ = list(map(int, input().split()))
    e.append(_[0])
    l.append(_[1])
    d.append(_[2])

time_matrix = []
for i in range(n + 1):
    _ = list(map(int, input().split()))
    time_matrix.append(_)


start_time = time.process_time()
e = [0] + e
l = [0] + l
d = [0] + d 

parents = []
best_fitness = 999999999
best_tour = []


def check_feasible(real_tỉme, p):
    real_time = 0
    for i in range(n):
        if i == 0: 
            real_time += d[0] + time_matrix[0][p[i]] # thời gian đén điểm i mà vượt quá khoảng l(i): false, break
            real_time +=  max(0,e[p[i]]-real_time)    # thời gian chờ nếu đến quá sớm
        else: 
            real_time += d[p[i-1]] + time_matrix[p[i-1]][p[i]] 
            real_time += max(0,e[p[i]]-real_time)
        if real_time not in range (e[p[i]], l[p[i]]):
            return False
    return True

def greedy_solution(n):              # by greedy
    global best_tour
    start_path_e = []
    greedy = sorted(e)
    for i in range (1, n + 1):
        for j in range (1, n + 1):
            if greedy[i] == e[j]:
                start_path_e.append(j)
                break
                
    start_path_l = []
    
    greedy = sorted(l)
    for i in range (1, n + 1):
        for j in range (1, n + 1):
            if greedy[i] == l[j]:
                start_path_l.append(j)
                break
    if check_feasible(0,start_path_e) and check_feasible(0,start_path_l) == False:
        best_tour = start_path_e
        return start_path_e
    elif check_feasible(0,start_path_l) and check_feasible(0,start_path_e) == False:
        best_tour = start_path_l
        return start_path_l
    else:
        best_tour = start_path_e
        return [start_path_e, start_path_l]
    
g_solution = greedy_solution(n)

def create_shuffle(n):
    lst = [i for i in range(1,n + 1)]
    random.shuffle(lst)
    return lst


def some_initial_population(n):
    # Create an initial population of random permutations of the cities
    cnt = 0
    while True :
        cnt += 1
        feasible_population = []
        for i in range(n):
            lst = create_shuffle(n)
            if check_feasible(0, lst):
                feasible_population.append(lst)
    
        ##################### Check Feasible #####################
        # if len(feasible_population) >= 2:
        #     break
        break
                
        #### quần thể không đủ lớn nên thêm tham lam vào cho đủ 
        
    dai = len(feasible_population)
    while dai < 4:
        if g_solution == None:
            return feasible_population
        else:
            if len(g_solution) != 2:
                feasible_population.append(g_solution)
                dai += 1
            elif len(g_solution) == 2:
                for i in range(2):
                    feasible_population.append(g_solution[i])
                    dai += 1
    if dai >= 4:
        return feasible_population
    # coi như đã có feasible solution
    return feasible_population
#bag = some_initial_population(n)
################# fitnesses #######################

'''
khi đã tạo được quần thể đầu tiên thì việc tiép theo là đánh giá mức độ phù hợp
để lựa chọn cặp gen trong việc xây dựng thế hệ tiếp theo
Việc này được thực hiện qua hàm EVALUATE()
'''
def fitness(tour):       # tour or chromosome, fitness or cost
    fitness = time_matrix[0][tour[0]]
    for i in range(n-1):
        fitness += time_matrix[tour[i]][tour[i + 1]]
    return fitness

def evaluate(bag):               # bag = some_feasible_solu
    fitnesses = [fitness(p) for p in bag]
    # fitnesses_copy = fitnesses.copy()
#    print(fitnesses)

    best_fitness = min(fitnesses)
    best_tour = bag[fitnesses.index(best_fitness)]
    parents.append(best_tour)

    sum_fitness = sum(fitnesses)
    max_fitness = max(fitnesses)

    
      # prevent GA premature converge, nếu [tour, tour,tour] không cái nào fitness khác nhau thì chọn cả, 
      #  còn nếu khác thì loại bỏ cái no hope
    boole = [fitnesses[0] == fitnesses[i] for i in range(n)]
#     print(a)
    # list_prob = []
    if False in boole :
#         for i in range(n):
#             fitnesses_copy[i] = max(fitnesses) - fitnesses_copy[i]
#         for i in range(n):    
#             list_prob.append(fitnesses_copy[i] / sum(fitnesses_copy)) # fitnessé bị thay đổi
# #         print(fitnesses_copy)
        

        list_prob = list(map(lambda x: x / (n*max_fitness - sum_fitness), fitnesses))


        return list_prob
    ### trường hợp tất cả tour có cost giống nhau
    # for i in range(n):
    #     list_prob.append(fitnesses[i] / sum(fitnesses))
    list_prob = list(map(lambda x: x/sum_fitness, fitnesses))
    return list_prob
  
   # return fitnesses, best_fitness, best_tour   

################# choose parents ######################


def select_parents(k):   # population_size
    global bag
    fit = evaluate(bag)
    while len(parents) < k:
        idx = random.randint(0,len(fit)-1)             # tìm idx từ 0 -> n-1
        if fit[idx] > random.random():
            parents.append(bag[idx])
    return parents


################ crucial part: mutation #####################
'''
ở phần mutation-crossover-swap sẽ là việc tháo, lắp các gen để tạo ra một gen tốt và hoàn chỉnh

'''

def swap(tour): # but very disruptive process in the context of TSP
    city1 = random.randint(0,n-1)      # từ thành phố thứ 2 đến n
    city2 = random.randint(0,n-1)

    while city1 == city2:
        city2 = random.randint(0, n-1)

    tour[city1], tour[city2] = tour[city2], tour[city1]
    return tour

# In crossover mutation, we grab two parents. 
# Then, we slice a portion of the chromosome of one parent,
# and fill the rest of the slots with that of the other parent.
# select_parents(n)

# no duplicate in chromosome
def crossover(p_cross):
    children = []
    count, size = len(parents), n
    for _ in range(len(bag)):
        if random.random() > p_cross:
            children.append(parents[random.randint(0, count-1)])
        else:
  ########################## chọn 2 cha mẹ để cấy ghép #######################
            a = random.randint(0, count - 1)
            b = random.randint(0, count - 1)
            while b == a:
                b = random.randint(0, count - 1)
            parent1, parent2 = parents[a], parents[b]
            
     ###################### bắt đầu cấy ghép ###############
            c = random.randint(0, size - 1)
            d = random.randint(0, size - 1)
            while d == c:
                d = random.randint(1, size - 1)
            if c<d:
                start, end = c, d
            else: start, end = d, c
                
        ######## phòng tránh bị lặp #########       
            child = [None] * (size)                 # fix None done
            for i in range(start, end + 1, 1):
                child[i] = parent1[i]
#           print(child, parent2)
            pointer = 0
            for i in range(size):                 
                if child[i] is None:
                    while parent2[pointer] in child:    
                        pointer += 1           
                    child[i] = parent2[pointer] 
            children.append(child)
    return children        

################  wrap the swap and crossover mutation into one nice function to call ##################33

def mutate(p_cross, p_mut):
    next_bag = []
    children = crossover(p_cross)
    for child in children:
        if random.random () < p_mut:
            next_bag.append(swap(child))
        else:
            next_bag.append(child)
    return next_bag
# bag = [generate_initial_tour(list_cities) for i in range(n)]
# print(bag)
# print(evaluate(bag))
# print(select_parents(4))

# A =(mutate(p_cross = 0.5, p_mut = 0.1))
# print(A)


'''
Gọi hàm TSPTW, điều chỉnh các thông số:
1.Num_generations: số lần chạy code để tạo quần thể mới...
2.Population_size: là tham số k trong hàm select_parents tượng trưng cho việc lấy k bố mẹ để tạo con.
                   qua thử nghiệm thì chỉ thấy K = 2 là đang ở mức ổn, số lớn hơn thì chạy rất lâu, thường là vô hạn time
3.P_crossocer, p_mut: các tham số để thuật toán chạy kết hợp với random rồi so sánh
                      Do chưa tìm thấy tài liệu nào nói về việc chọn 2 loại xác suất này 
                      Nên tôi lấy theo Bard - AI khuyên:
                          Đối với các quần thể lớn, p_crossover có thể được giảm xuống để tránh trùng lặp.
                          Nếu các giải pháp tốt thường được tìm thấy, thì p_mutation có thể được giảm xuống để tránh mất chúng.
                      p_mutation: 0.01-0.1
                      p_crossover: 0.6-0.9
'''

def tsptw_ga(n, num_generations, population_size , p_crossover, p_mut, end_time):
    
    global best_tour, best_fitness, parents
    bag = some_initial_population(n)

    if len(bag) == 0:
        return []    
    fitness_1st_bag = [fitness(i) for i in bag]
    min_fitness =  min(fitness_1st_bag)
    index_min_fitness = fitness_1st_bag.index(min_fitness)
    if min_fitness < best_fitness:
        best_tour = bag[index_min_fitness]

    # Run the genetic algorithm for a specified number of generations
    for _ in range(num_generations):
        if time.process_time() > end_time:
            return

        try:
            a = min([fitness(i) for i in bag])
            if a <= best_fitness:
                best_fitness = a
            # Evaluate the fitness of each individual in the population

            # Select parents based on their fitness
            parents = select_parents(population_size)
            # Create offspring using crossover and mutation
            children = []
            children = mutate(p_crossover, p_mut)

            # Replace the old population with the new population
            bag = children
            if min([fitness(p) for p in bag]) < best_fitness:
                best_fitness = min([fitness(p) for p in bag])
                best_tour = p
        except:
            continue
    return
print(n)
def main():
    global start_time
    if n <=100:
        end_time = start_time + 10
    else: 
        end_time = start_time + 280
    end_time = end_time = start_time + 180
    while True:
        tsptw_ga(n, num_generations = 20,  population_size = 10, p_crossover = 0.9, p_mut = 0.09, end_time=end_time)
        if time.process_time() > end_time:
            break
    # tsptw_ga(n, num_generations = 5,  population_size = 10, p_crossover = 0.9, p_mut = 0.09, end_time=end_time)

print(g_solution)
main()
print(best_fitness)
print(*best_tour)
# Nếu là test chặt thì khó tránh khỏi việc chỉ có thể ra greedy nếu nó tồn tại

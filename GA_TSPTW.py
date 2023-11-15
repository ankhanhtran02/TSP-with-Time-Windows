import random

### mn thay tên file là ra đáp án nhé, còn điều chỉnh tham số thì ở dòng tsptw_ga ( 268)

def read_input_file(filename='TSPTW_test_2.txt'):
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
    
    
n , e, l, d, time_matrix = read_input_file('TSPTW_test_2.txt')
e = [0] + e
l = [0] + l
d = [0] + d 

parents = []
best_fitness = 0
best_tour = []


# greedy = sorted(l)
# def greedy_solution(n):              # by greedy
#     start_path = []
#     for i in range (1, n + 1):
#         for j in range (1, n + 1):
#             if greedy[i] == l[j]:
#                 start_path.append(j)
#                 break
#     return start_path 

greedy = sorted(e)
def greedy_solution(n):              # by greedy
    start_path = []
    for i in range (1, n + 1):
        for j in range (1, n + 1):
            if greedy[i] == e[j]:
                start_path.append(j)
                break
    return start_path 

def create_shuffle(n):
    lst = [i for i in range(1,n + 1)]
    random.shuffle(lst)
    return lst

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
def some_initial_population(n):
    # Create an initial population of random permutations of the cities
    cnt = 0
    while True :
        cnt += 1
        population = []
        for i in range(n):
            lst = create_shuffle(n)
            population.append(lst)
        ##################### Check Feasible #####################
        feasible_population = []
        for p in population:
            is_feasible = True
            real_time = 0
            is_feasible = check_feasible(real_time, p)
            if is_feasible:
                feasible_population.append(p)
                
        #### quần thể khôgn đủ lớn nên thêm tham lam vào cho đủ 
        
        if cnt >= n:
            while len(feasible_population) < n:
                if check_feasible(0, greedy_solution(n)) == False:
                    return feasible_population
                feasible_population.append(greedy_solution(n))

        if len(feasible_population) >= n:
            break
    # coi như đã có feasible solution
    return feasible_population
#bag = some_initial_population(n)
################# fitnesses #######################
def fitness(tour):       # or chromosome, fitness or cost
    fitness = time_matrix[0][tour[0]]
    for i in range(n-1):
        fitness += time_matrix[tour[i]][tour[i + 1]]
    return fitness

def evaluate(bag):               # bag = some_feasible_solu
    fitnesses = [fitness(p) for p in bag]
    fitnesses_copy = fitnesses.copy()
#    print(fitnesses)

    best_fitness = min(fitnesses)
    best_tour = bag[fitnesses.index(best_fitness)]
    parents.append(best_tour)
    
      # prevent GA premature converge, nếu [tour, tour,tour] không cái nào fitness khác nhau thì chọn cả, 
      #  còn nếu khác thì loại bỏ cái no hope
    
    boole = [fitnesses[0] == fitnesses[i] for i in range(n)]
#     print(a)
    list_prob = []
    if False in boole :
        for i in range(n):
            fitnesses_copy[i] = max(fitnesses) - fitnesses_copy[i]
        for i in range(n):    
            list_prob.append(fitnesses_copy[i] / sum(fitnesses_copy)) # fitnessé bij thay đổi
#         print(fitnesses_copy)
        
        return list_prob
    ### trường hợp tất cả tour có cost giống nhau
    for i in range(n):
        list_prob.append(fitnesses[i] / sum(fitnesses))
    return list_prob
  
   # return fitnesses, best_fitness, best_tour   

################# choose parents ######################

def select_parents(k):   # population_size
    fit = evaluate(bag)
    while len(parents) < k:
        idx = random.randint(0,len(fit)-1)             # tìm idx từ 0 -> n-1
        if fit[idx] > random.random():
            parents.append(bag[idx])
    return parents


################ crucial part: mutation #####################

def swap(tour): # but very disruptive process in the context of TSP
    city1 = random.randint(0,n-1)      # từ thành phố thứ 2 đến n
    city2 = random.randint(0,n-1)
    
    while city1 == city2:
        city2 = random.randint(0, n-1)
        
    new_tour = tour.copy()
    new_tour[city1], new_tour[city2] = new_tour[city2], new_tour[city1]
    return new_tour

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




def tsptw_ga(n, num_generations, population_size , p_crossover, p_mut):
    bag = some_initial_population(n)
    
    ####        quần thể không có ai   #######
    if len(bag) == 0:
        return []     
    
    
    global best_tour, best_fitness, parents
    best_tour = bag[0]
    # Run the genetic algorithm for a specified number of generations
    for _ in range(num_generations):
        try:
            best_fitness = min([fitness(i) for i in bag])
            # Evaluate the fitness of each individual in the population
            fitnesses = evaluate(bag)
            # Select parents based on their fitness
            parents = select_parents(population_size)
            # Create offspring using crossover and mutation
            children = []
            children = mutate(p_cross, p_mut)

            # Replace the old population with the new population
            bag = children
            if min([fitness(i) for i in bag]) < best_fitness:
                best_fitness = min([fitness(i) for i in bag])
                best_tour = i
            
        except:
            continue
    return best_tour
print(n)
best_tour = tsptw_ga(n, num_generations = 50,  population_size = 20, p_crossover = 0.9, p_mut = 0.09)
print(*best_tour)
print(best_fitness)

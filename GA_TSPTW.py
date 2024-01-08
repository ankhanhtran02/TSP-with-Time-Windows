import random

import time

######### Input ############
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


parents = []
best_fitness = 999999999
best_tour = []


def check_feasible(real_tỉme, p):
    # try except to prevent p is None
    try:
        real_time = 0
        for i in range(n):
            if i == 0: 
                real_time += d[0] + time_matrix[0][p[i]] # time passed right before service at i is not pass l[i]
                real_time +=  max(0,e[p[i]]-real_time)    # bonus waiting time if salesman appear at i to soon
            else: 
                real_time += d[p[i-1]] + time_matrix[p[i-1]][p[i]] 
                real_time += max(0,e[p[i]]-real_time)
            if real_time not in range (e[p[i]], l[p[i]]):
                return False
        return True
    except:
        return False

def greedy_solution(n):
    ''' Greedy solution sorted by starting_time or ending_time '''
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
    

'''Initial population will be create by random shuffle'''
def create_shuffle(n):
    lst = [i for i in range(1,n + 1)]
    random.shuffle(lst)
    return lst

'''This function create the initial population'''
def some_initial_population(n):
    # Create an initial population of random permutations of the cities
    cnt = 0
    while True :
        cnt += 1
        feasible_population = []
        for i in range(n):
            lst = create_shuffle(n)
            if check_feasible(0, lst):       # check if the path is feasible 
                feasible_population.append(lst)
        break
                
    # the population might be short so add greedy solution to it 
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
    return feasible_population



''' 
After having the initial population, next step is evaluating the fitnesses to 
select the parents to generate next generation. This is done by Evaluate function      
'''
def fitness(tour):       # tour or chromosome, fitness or total time traveling of each path
    fitness = time_matrix[0][tour[0]]
    for i in range(n-1):
        fitness += time_matrix[tour[i]][tour[i + 1]]
    return fitness

def evaluate(bag):             
    fitnesses = [fitness(p) for p in bag]
    best_fitness = min(fitnesses)
    best_tour = bag[fitnesses.index(best_fitness)]
    parents.append(best_tour)

    sum_fitness = sum(fitnesses)
    max_fitness = max(fitnesses)

    # prevent GA premature converge, if [tour, tour,tour] all have the same fitness so we select all, 
    # if they are different, so wak individuals will be eliminate
    
    boole = [fitnesses[0] == fitnesses[i] for i in range(n)]
    if False in boole :
        list_prob = list(map(lambda x: x / (n*max_fitness - sum_fitness), fitnesses))
        return list_prob
    list_prob = list(map(lambda x: x/sum_fitness, fitnesses))
    return list_prob
  

'''This function will select parents base on fitnesses calculate by Evaluate function'''
def select_parents(k):   # k: population_size
    global bag
    fit = evaluate(bag)
    while len(parents) < k:
        idx = random.randint(0,len(fit)-1)             # find index from 0 to n-1
        if fit[idx] > random.random():
            parents.append(bag[idx])
    return parents


'''
Next Step is the most crucial part: Mutation
mutation has crossover-swap is the way we remove, add, 
transfrom chromosomes to create a child eficient end good
'''

'''Swap: change the location of two vertices'''
def swap(tour): # but very disruptive process in the context of TSP
    city1 = random.randint(0,n-1)      
    city2 = random.randint(0,n-1)
    while city1 == city2:
        city2 = random.randint(0, n-1)
    tour[city1], tour[city2] = tour[city2], tour[city1]
    return tour

'''
In crossover mutation, we grab two parents. 
Then, we slice a part of the chromosome of one parent,
and fill the rest of the slots with that of the other parent.
select_parents(n)
So that there will be no repetition of vertices
'''

def crossover(p_cross):
    children = []
    count, size = len(parents), n
    for _ in range(len(bag)):
        if random.random() > p_cross:
            children.append(parents[random.randint(0, count-1)])
        # select two parents to cross
        else:
            a = random.randint(0, count - 1)
            b = random.randint(0, count - 1)
            while b == a:
                b = random.randint(0, count - 1)
            parent1, parent2 = parents[a], parents[b]
            
            # start cross
            # slice to get one part ò a parent
            c = random.randint(0, size - 1)
            d = random.randint(0, size - 1)
            while d == c:
                d = random.randint(1, size - 1)
            if c<d:
                start, end = c, d
            else: start, end = d, c
                
            # prevent repetition      
            child = [None] * (size)                
            for i in range(start, end + 1, 1):
                child[i] = parent1[i]
            pointer = 0
            for i in range(size):                 
                if child[i] is None:
                    while parent2[pointer] in child:    
                        pointer += 1           
                    child[i] = parent2[pointer] 
            children.append(child)
    return children        

'''
wrap the swap and crossover  into one nice function call mutation
(the name of this function is misunderstanding, you can just define that is just 
some name to wrap 2 above function)to call 
'''

def mutate(p_cross, p_mut):
    next_bag = []
    children = crossover(p_cross)
    for child in children:
        if random.random () < p_mut:
            next_bag.append(swap(child))
        else:
            next_bag.append(child)
    return next_bag


'''
TSPTW function, change the parameters:

1.Num_generations: number of generation, with strict constraints, this should be small
2.Population_size: is the parameter k in the select_parents function that represents taking k parents to create children.
3.P_crossocer, p_mut: parameters for the algorithm to run in combination with random and then compare
                      Because I have not found any documents talking about choosing these two types of probabilities
                      So I took what I saw at TSP:
                          For large populations, p_crossover can be reduced to avoid overlap.
                          If good solutions are often found, then p_mutation can be reduced to avoid losing them.
                      p_mutation: 0.01-0.1
                      p_crossover: 0.6-0.9
'''

time_update = []
g_solution = greedy_solution(n)

def tsptw_ga(n, num_generations, population_size , p_crossover, p_mut, end_time):
    
    global best_tour, best_fitness, parents, time_update
    bag = some_initial_population(n)

    if len(bag) == 0:
        return []    
    fitness_1st_bag = [fitness(i) for i in bag]
    min_fitness =  min(fitness_1st_bag)
    index_min_fitness = fitness_1st_bag.index(min_fitness)
    if min_fitness < best_fitness:
        best_fitness = min_fitness
        best_tour = bag[index_min_fitness]
        time_update.append(time.process_time())

    # Run the genetic algorithm for a specified number of generations
    for _ in range(num_generations):
        if time.process_time() > end_time:
            return

        try:
            a = min([fitness(i) for i in bag])
            if a < best_fitness:
                best_fitness = a
                time_update.append(time.process_time())
            # Evaluate the fitness of each individual in the population

            # Select parents based on their fitness
            parents = select_parents(population_size)
            # Create offspring using crossover and mutation
            children = []
            children = mutate(p_crossover, p_mut)

            # Replace the old population with the new population
            bag = children
            lf = [fitness(p) for p in bag]
            mn = min(lf)
            if mn < best_fitness:
                best_fitness = mn
                best_tour = bag[lf.index(mn)]
                time_update.append(time.process_time())
        except:
            continue
    return

def main():
    global start_time
    # set the proper time to run algorithm
    end_time = end_time = start_time + 180
    while True:
        tsptw_ga(n, num_generations = 30,  population_size = 10, p_crossover = 0.9, p_mut = 0.09, end_time=end_time)
        if time.process_time() > end_time:
            break
print(n)
main()
# print('time_update:', time_update)
# print('time:',time_update[-1])
# print(best_fitness)
print(*best_tour)

import random
import numpy as np
import time
from test_generator import read_input_file, TSPTW_cost, convert_no_D_inputs
from urllib.request import urlopen
from os.path import exists
from collections import deque


def online_inp(file_name):
    '''
    This function reads the input file from the online github link provided by our team, or reads from local file if a file with the same name already exists
    '''
    if exists(file_name):
        print('Reading from existing input file')
        return read_input_file(file_name)
    base_URL = "https://raw.githubusercontent.com/ankhanhtran02/TSP-with-Time-Windows/main/test_cases/"
    test_file_name = file_name
    txt = urlopen(base_URL + test_file_name).read().decode("utf-8")
    with open(file_name, "w") as file:
        file.write(txt)
    return read_input_file(file_name)

class Node:
    '''
    - Node is the basic object to build the Tree object for the Probabilistic Beam Algorithm
    - The class variable h_params, when the first instance of Node is created, will be calculated and have a value of a tuple containing:
        + h_params[0]: a tuple (N, e, l, d, t) represents the input of the TSPTW problem
        + h_params[1]: a tuple (min(e), max(e), min(l), max(l), min(t), max(t)) containing the minimum and maximum value of each array e, l and t for calculating the heuristic value
        + h_params[2]: a tuple (lambda_e, lambda_l, lambda_c) containing the weights of the heuristic for the lower bound, upper bound of the time window and the travel time between two nodes
        + h_params[3]: the pheromone matrix for the ACO algorithm, this is the only heuristic parameters that will change in runtime
    '''
    h_params = (None, None, None, None)
    def __init__(self, value, parent, problem_inputs=None):
        self.Children = [] # list of child nodes of the current nodes
        self.extensions = None # Set of nodes that can be extend from this node
        if value == 0: # Initialize the extensions set for the root node of the tree
            self.extensions = set(range(1, problem_inputs[0] + 1))
        self.value = value # The value ranging from 0 (only the first node of the tree) to n
        self.parent = parent # The parent Node
        if Node.h_params[0] == None: # Initialize the heuristic parameters for the class when creating the first Node
            self.calculate_h_params(problem_inputs)
        self.h_inf = (None, 0, 0, 0) # The heuristic information asscociated with choosing this node (pheromone*nuy, rank_sum, times_passed, violations_count)
    def AddChild(self, child_value, heuristic_tuple):
        child = Node(child_value, self)
        child.extensions = self.extensions - set([child_value])
        child.h_inf = heuristic_tuple
        self.Children.append(child)

    def CalculateRankSum(self):
        '''
        Calculate the cumulated rank of each children of this Node, based on how good its heuristic is compared to its siblings
        This will modify the second value in the h_inf attribute of each children
        '''
        if len(self.Children) == 0:
            return
        ranks = sorted([child for child in self.Children], key=lambda item:item.h_inf[0])
        for i in range(len(ranks)-1, -1, -1):
            ranks[i].h_inf = list(ranks[i].h_inf)
            ranks[i].h_inf[1] = self.h_inf[1] + (len(ranks) - i)
            ranks[i].h_inf = tuple(ranks[i].h_inf)
    def __str__(self):
        return f'value({self.value}):h_inf({self.h_inf})'

    def calculate_h_params(self, problem_inputs:tuple): 
        '''
        Calculate the parameters necessary for heuristic calculation
        '''
        N, e, l, d, t = problem_inputs
        # lambdas = [random.random() for _ in range(3)]
        # lambda_c, lambda_l, lambda_e = [item/sum(lambdas) for item in lambdas]
        lambda_c, lambda_l, lambda_e = 0.75, 0.1, 0.15
        min_t = 99999999999999999999999
        max_t = -99999999999999999999999
        min_e, max_e, min_l, max_l = min(e), max(e), min(l), max(l)
        lambda_heuristics = [[0 for i in range(N+1)] for j in range(N+1)]
        for i in range(N+1):
            for j in range(N+1):
                if t[i][j] < min_t:
                    min_t = t[i][j]
                if t[i][j] > max_t:
                    max_t = t[i][j]
        for i in range(N+1):
            for j in range(N+1):
                lambda_heuristics[i][j] = ((max_e - e[j-1])/(max_e - min_e), (max_l - l[j-1])/(max_l - min_l), (max_t - t[i][j])/(max_t - min_t))
        e_temp = [0] + e
        l_temp = [0] + l
        proximity = [[min(l_temp[j], l_temp[i] + t[i][j]) - max(e_temp[j], e_temp[i] + t[i][j]) for j in range(N+1)] for i in range(N+1)]

        Node.h_params = ((N, e, l, d, t), (min_e, max_e, min_l, max_l, min_t, max_t), (lambda_e, lambda_l, lambda_c), (np.ones((N+1, N+1)) * 0.1, lambda_heuristics, proximity))

def calculate_heuristic(parent:Node, new_value:int):
    '''
    This function calculates a tuple containing the heuristic value for a new Node when it is added to the tree, provided the information of the parent Node and the value of the new Node
    '''
    assert isinstance(new_value, int)
    assert Node.h_params != None
    p = Node.h_params
    # Node.h_params will be a tuple containing:
    # index 0: (N, e, l, d, t)
    # index 1: (min(e), max(e), min(l), max(l), min(t), max(t))
    # index 2: (lambda_e, lambda_l, lambda_c)
    # index 3: ((N+1) x (N+1) pheromone matrix, lambda_heuristics, proximity)
    assert isinstance(p[0][0], int)
    assert len(p[0]) == 5
    assert len(p[1]) == 6
    assert len(p[2]) == 3
    assert isinstance(p[1][5], int)
    assert isinstance(p[0][1][new_value - 1], int)
    if parent == None: # When the first Node of the Tree is created, there is no further information
        return 0
    nuy = p[2][2] * p[3][1][parent.value][new_value][2] + p[2][1] * p[3][1][parent.value][new_value][1] + p[2][0] * p[3][1][parent.value][new_value][0]
    pheromone = p[3][0][parent.value][new_value]
    times_passed = max((parent.h_inf[2] + p[0][4][parent.value][new_value]), p[0][1][new_value - 1]) # The total time passed before getting to the point new_value
    v = 0 # Check if putting the new node in the path creates a new violation
    # print(times_passed)
    if times_passed >= p[0][2][new_value - 1]:
        v = 1
    # return (pheromone*nuy*int(p[3][2][parent.value][new_value] > -500)+0.0001, 0, times_passed, parent.h_inf[3] + v)
    return (max(p[3][2][parent.value][new_value], 100) * pheromone, 0, times_passed, parent.h_inf[3] + v)

def lex_compare(leave_node1:Node, leave_node2:Node) -> bool:
    '''
    This is a compare function for two leave nodes of the Tree object, so we can update the best leave node so far
    '''
    assert Node.h_params[0]!= None
    if leave_node2 == None:
        return True
    if leave_node1.h_inf[3] < leave_node2.h_inf[3]:
        return True
    elif leave_node1.h_inf[3] == leave_node2.h_inf[3]:
        if leave_node1.h_inf[3] == 0:
            return TSPTW_cost(Node.h_params[0], path_from_leave(leave_node1)) < TSPTW_cost(Node.h_params[0],path_from_leave(leave_node2))
        else:
            cost1, cost2 = 0, 0
            t = Node.h_params[0][4]
            path1 = path_from_leave(leave_node1)
            path2 = path_from_leave(leave_node2)
            for i in range(0, Node.h_params[0][0] - 1):
                cost1 += t[path1[i]][path1[i+1]]
                cost2 += t[path2[i]][path2[i+1]]
            return cost1 < cost2
    else:
        return False
      
def path_from_leave(leave:Node):
    '''
    This function takes a leave Node as a parameter and return a list containing the path from the tree root to that Node by tracing the parent of the Node until we get to the root
    '''
    pointer = leave
    temp_path = [pointer.value]
    while pointer.parent != None:
        temp_path.append(pointer.parent.value)
        pointer = pointer.parent
    return temp_path[::-1][1:]

class Tree:
    '''
    - The main object for the Probabilistic Beam Solver
    - Every instance of this class will have 3 main parameters for the PBS:
        + k_bw (int): this is the beam width: the maximum number of nodes kept after each shrinking step for further expansion
        + muy (float): the expansion rate. If there are currently L leave nodes, after the expansion step it will be L*muy
        + N_s (int): the number of stochastic samples taken for partial evaluation
    '''
    def __init__(self, problem_inputs, k_bw, muy, N_s):
        self.root = Node(0, None, problem_inputs=problem_inputs)
        self.leaves = [self.root] # This list will store the leave nodes of the tree after every step
        self.heuristics = [] # (Node, new, h) This list stores the calculated heuristics after each expansion step
        self.k_bw = k_bw
        self.muy = muy
        self.N_s = N_s
        self.strict = True # This guarantee any path generated doesn't violate time constraints. If every path found after expansion violates time window constraints, then this will automatically be set to False
    
    def reset_leaves(self):
        for leave in self.leaves:
            leave.Children = []
            leave.extensions = {}

    def expand(self, q0=0.5):
        '''
        - Expand the tree by adding more children to the current leave nodes, after calculating the heuristic.
        - q0 is the probability that the children are chosen deterministically, meaning the nodes with the highest heuristic values
        - 1-q0 is the probability that the children are chosen randomly with their heuristic values being the weights
        '''
        self.heuristics = []
        for leave_node in self.leaves:
            for new in leave_node.extensions:
                self.heuristics.append((leave_node, new, calculate_heuristic(leave_node, new)))
        if self.strict:
            print(f'current len is: {len(self.heuristics)}')
            self.heuristics = [item for item in self.heuristics if (item[2][0] > 0 and item[2][3] == 0)]
            print(len(self.heuristics))
            if len(self.heuristics) == 0:
                self.strict = False
                self.expand()
                return
        else:
            self.heuristics = [item for item in self.heuristics if (item[2][0] > 0)]
        self.heuristics.sort(key=lambda item:item[2][0]) # Sort by the product of the pheromone*nuy function
        new_leaves = []
        q = random.random()
        if q < q0:
            # print("Expanding deterministically")
            # Select the best k_bw new possible nodes to expand
            for i in range(-min(int(self.muy * len(self.leaves))+1, len(self.heuristics)), 0):
                self.heuristics[i][0].AddChild(self.heuristics[i][1], self.heuristics[i][2]) # Add a child to the best leave nodes
                new_leaves.append(self.heuristics[i][0].Children[-1]) # Add the previously added child to the list of new leaves

        else:
            # print("Expanding using weights for random choices")
            # Select from heuristics list extensions with corresponding weights, no replacement
            S = sum([item[2][0] for item in self.heuristics])
            chosen = np.random.choice(len(self.heuristics), p=[item[2][0]/S for item in self.heuristics], size=max(min(int(self.muy * len(self.leaves))+1, len(self.heuristics)), 3*len(self.heuristics)//4), replace=False)
            temp = []
            for choice in chosen:
                temp.append(self.heuristics[choice])
            chosen = temp

            for extension in chosen:
                extension[0].AddChild(extension[1], extension[2]) # Add a child to the best leave nodes
                new_leaves.append(extension[0].Children[-1]) # Add the previously added child to the list of new leaves
        for leave in self.leaves:
            leave.CalculateRankSum()
        self.reset_leaves()
        self.leaves = new_leaves
        # Modify the leaves array
    def shrink(self):
        '''
        Sort the leaves in ascending order of the number of violations and if they are equal, sort according to the sum rank and only keep k_bw best leaves for future expansion
        '''
        self.leaves.sort(key=lambda item:(item.h_inf[3], item.h_inf[2], item.h_inf[1]))
        self.leaves = self.leaves[:self.k_bw]
    def __str__(self):
        return ''

class BeamSolver:
    '''
    - The Solver object which uses the Probabilistic Beam Algorithm
    - Every instance of this class will have 3 main parameters for the PBS:
        + k_bw (int): this is the beam width: the maximum number of nodes kept after each shrinking step for further expansion
        + muy (float): the expansion rate. If there are currently L leave nodes, after the expansion step it will be L*muy
        + N_s (int): the number of stochastic samples taken for partial evaluation
    '''
    def __init__(self, k_bw, muy, N_s, q0=0.5):
        self.k_bw = k_bw
        self.muy = muy
        self.N_s = N_s
        self.found_paths = []
        self.q0 = q0
    def fit(self, problem_inputs):
        n, e, l, d, t = problem_inputs
        self.solver_tree = Tree(problem_inputs, self.k_bw, self.muy, self.N_s)
        for i in range(1, n+1):
            self.solver_tree.expand(self.q0)
            self.solver_tree.shrink()
        # print(self.solver_tree)
        # print('The final leaves of the trees are:')
        # for leave in self.solver_tree.leaves:
        #     print(leave)
        for leave in self.solver_tree.leaves:
            self.found_paths.append(path_from_leave(leave))

def ApplyPheromoneUpdate(pb_iter, pb_restart, pb_bf, ro=0.1, K_iter=0.2, K_restart=0.3, K_bf=0.5):
    '''
    - Given the best path in the iteration, best path after restarting the algorithm and the best path found so far, this function updates the pheromone matrix contained in the Node.h_params variable according to the weights specified by K_iter, K_restart and K_bf, respectively
    - ro is the update coefficient, the larger ro is, the faster the matrix will be updated
    '''
    current_pm = Node.h_params[3][0]
    N = Node.h_params[0][0]
    assert len(current_pm) == N + 1
    assert len(pb_iter) == N and len(pb_bf) == N and len(pb_restart) == N
    for i in range(0, N - 1):
        current_pm[pb_iter[i]][pb_iter[i+1]] += ro*K_iter
        current_pm[pb_restart[i]][pb_restart[i+1]] += ro*K_restart
        current_pm[pb_bf[i]][pb_bf[i+1]] += ro*K_bf
    return

def Calculate(path):
    n, e, l, d, t = Node.h_params[0]
    cur_pos = 0
    total_time = 0
    travel_time = 0
    for i in range (0, n):
        total_time += t[cur_pos][path[i]]
        travel_time += t[cur_pos][path[i]]
        if total_time <= l[path[i]-1]:   
            total_time = max(total_time, e[path[i]-1])
            total_time += d[path[i]-1]
            cur_pos = path[i]
        else: 
            return 99999999999
    return travel_time

def Tabu(start_path:list, run_time=10):
    '''
    Tabu search, which will be used as a local search algorithm to enhance the path found by ACO and the best path found so far
    '''
    problem_inputs = Node.h_params[0]
    start_time = time.time()
    n, e, l, d, t = problem_inputs
    e = [-1] + e
    l = [-1] + l
    d = [-1] + d
    tabu = deque()
    tabu.append((1, start_path))

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

    cur_MIN = Calculate(start_path)
    list_optimal_path = [start_path]
    cur_OPTIMAL_path = start_path
    while True: 
        element = tabu.popleft()  # This pops the first inserted item
        if element[0] == 0: 
            list_optimal_path.append(element)
            tabu.append((1, element[1])) # continue to search the current optimal solution
        elif element[0] <= 3:
            tabu.append(element)
            cur_best_element = []
            found_better = False
            for i in range (0, n - 1):
                for j in range (i + 1, min(i + 5, n)):
                    try_test = list(element) # get the path
                    try_path = try_test[1][:]
                    try_path[i], try_path[j] = try_path[j], try_path[i]
                    if Calculate(try_path) < cur_MIN:
                        found_better = True # found a local maximum
                        cur_best_element = []
                        cur_best_element.append((0, try_path))
                        cur_OPTIMAL_path = try_path[:]
                        cur_MIN = Calculate(try_path)
                        print("Time taken", time.time() - start_time, "CUR_MIN", cur_MIN)
                    elif Calculate(try_path) == cur_MIN:
                        cur_best_element.append((0, try_path))
                    cur_best_element.append((try_test[0] + 1, try_path))
                    if time.time() - start_time > run_time: 
                        break
            if found_better == True and len(cur_best_element) > 0:
                tabu.clear()
            for i in cur_best_element:
                tabu.append(i)
        if time.time() - start_time > run_time: 
            break

    
    for path in list_optimal_path:
        if path[0] == 0 and Calculate(path[1]) <= cur_MIN:
            cur_MIN = Calculate(path[1])
            cur_OPTIMAL_path = path[1]
    e = e[1:]
    l = l[1:]
    d = d[1:]
    return cur_OPTIMAL_path

class ACOSolver:
    '''
    - The main Solver class that implements ACO.
    - The parameters k_bw, muy, N_s, q0 of the Probabilistic Beam Search will be passed to the BeamSolver object

    '''
    best_10 = []
    def __init__(self, k_bw:int, muy:float, N_s:int, q0):
        self.N_s = N_s
        self.k_bw = k_bw
        self.muy = muy
        self.q0 = q0
        self.leave_P_bf = None
        self.leave_P_rb = None
        self.cf = 0 # convergence factor
        self.bs_update = False
        self.solution_time = 300 # The time that the algorithm is allowed to run, 300 seconds by default
        self.best_tabu_path = None
        self.setParameters() # Set necessary parameters for the ACO algorithm
    def setParameters(self, ro=0.1, K_iter=0.2, K_restart=0.3, K_bf = 0.5, X=10):
        '''
        Set the parameters for the ACO framework
        ro is the learning rate that determines how fast the pheromone update is
        K_iter, K_restart and K_bf are the weight of the best solution during each solution, each restart and the best solution found so far used during pheromone matrix update
        X is the number of iterations after which the algorithm restarts
        '''
        assert K_iter + K_restart + K_bf == 1
        self.X = X
        self.ro = ro
        self.K_iter = K_iter
        self.K_restart = K_restart
        self.K_bf = K_bf
        
    def setSolutionTime(self, solution_time:int):
        assert solution_time > 0 and solution_time <= 300
        self.solution_time = solution_time
    def fit(self, problem_inputs:tuple):
        '''
        - Feed the input of the TSPTW problem to the Solver
        - problem_inputs as always will be a tuple (N, e, l, d, t) representing the problem
        '''
        self.problem_inputs = problem_inputs
        assert len(problem_inputs) == 5
        if Node.h_params[3] != None:
            print(Node.h_params)
            assert False
        path_greedy_e = [x[0] for x in sorted([(i+1, problem_inputs[1][i]) for i in range(problem_inputs[0])], key=lambda item:item[1])]
        print(TSPTW_cost(problem_inputs, path_greedy_e))
        self.best_path = path_greedy_e
        
        # Initialize a trash Node object to reset the h_params variable of class Node
        trash_node = Node(0, None, problem_inputs)
        trash_node.calculate_h_params(problem_inputs)

        assert len(Node.h_params[3][0]) == problem_inputs[0] + 1 and len(Node.h_params[3][0][0]) == problem_inputs[0] + 1
        start_loop = time.time()
        iteration = 0

    
        while True:
            beamSolver = BeamSolver(self.k_bw, self.muy, self.N_s, self.q0)
            beamSolver.fit(self.problem_inputs)
            leave_P_ib = beamSolver.solver_tree.leaves[0] # Take the best leave node returned by the BeamSolver
            print('number of ant violations: ' + str(leave_P_ib.h_inf[3]))
            if lex_compare(leave_P_ib, self.leave_P_rb):
                self.leave_P_rb = leave_P_ib
            if lex_compare(leave_P_ib, self.leave_P_bf):
                self.leave_P_bf = leave_P_ib
            if iteration == 0:
                if TSPTW_cost(problem_inputs, path_from_leave(self.leave_P_bf))!=None and TSPTW_cost(problem_inputs, path_from_leave(self.leave_P_bf)) < TSPTW_cost(problem_inputs, self.best_path):
                    self.best_path = path_from_leave(self.leave_P_bf)
                    print(f'new cost is {TSPTW_cost(problem_inputs, self.best_path)} iteration 0')
            # cf = ComputeCF(Node.h_params[3][0]) # Calculate the convergence factor of the pheromone matrix
            if self.bs_update and iteration % self.X == 0:
                # trash_node = Node(0, None, problem_inputs)
                # trash_node.calculate_h_params(self.problem_inputs)
                self.leave_P_rb = None
                self.bs_update = False
            else:
                if iteration % self.X == 0:
                    print(f'Current iteration is #{iteration}')
                    self.bs_update = True
                    iteration -= 1
                    self.best_path = Tabu(self.best_path, 3)
                    print(f'new cost is {TSPTW_cost(problem_inputs, self.best_path)} tabu')
                self.best_tabu_path = path_from_leave(self.leave_P_rb)
                if iteration % self.X <= 5:
                    self.best_tabu_path = Tabu(path_from_leave(self.leave_P_rb), 1)
                temp_best = path_from_leave(self.leave_P_bf)
                if TSPTW_cost(self.problem_inputs, temp_best) == None or TSPTW_cost(self.problem_inputs, self.best_tabu_path) < TSPTW_cost(self.problem_inputs, temp_best):
                    temp_best = self.best_tabu_path
                assert len(temp_best) == self.problem_inputs[0]

                if TSPTW_cost(self.problem_inputs, temp_best) != None and ((TSPTW_cost(self.problem_inputs, self.best_path) == None or TSPTW_cost(self.problem_inputs, temp_best) < TSPTW_cost(self.problem_inputs, self.best_path))):
                    self.best_path = temp_best[:]
                    print(f'new cost is {TSPTW_cost(problem_inputs, self.best_path)} ant')
                else:
                    print(TSPTW_cost(problem_inputs, temp_best))
                
                ApplyPheromoneUpdate(path_from_leave(leave_P_ib), temp_best, self.best_path,self.ro, self.K_iter, self.K_restart, self.K_bf)
                # ApplyPheromoneUpdate(self.best_path, temp_best, self.best_path,self.ro, self.K_iter, self.K_restart, self.K_bf)
                # print(Node.h_params[3][0])
            end_current_loop = time.time()
            if end_current_loop - start_loop > self.solution_time:
                break
            iteration += 1

if __name__ == "__main__":
    problem_inputs = online_inp('new_test_cases/B30_2.txt')
    k_bw, muy, N_s = 200, 30, 3
    ro, K_iter, K_restart, K_bf, X = 0.9, 0.05, 0.3, 0.65, 5
    main_solver = ACOSolver(k_bw, muy, N_s, 0.9)
    main_solver.setSolutionTime(180)
    main_solver.setParameters(ro, K_iter, K_restart, K_bf, X)
    
    # main_solver = BeamSolver(k_bw, muy, N_s, 1)
    main_solver.fit(problem_inputs)
    print(f'Best path found so far is: {main_solver.best_path}')
    print(f'The cost of the best path found is {TSPTW_cost(problem_inputs, main_solver.best_path)}')
    # cur_min = 999999999999999
    # for path in main_solver.found_paths:
    #     a = TSPTW_cost(problem_inputs, path)
    #     if a != None and a < cur_min:
    #         cur_min = a
    # print(cur_min)
        
    # print(main_solver.found_paths[0], TSPTW_cost(problem_inputs, main_solver.found_paths[0]))
    # print(main_solver.found_paths[-1], TSPTW_cost(problem_inputs, main_solver.found_paths[1]))


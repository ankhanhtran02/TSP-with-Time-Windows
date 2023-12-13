from test_generator import TSPTW_cost, read_input_file
import os

class Node:
    def __init__(self, ID, e, l, d):
        self.ID = ID
        self.e = e
        self.l = l
        self.d = d
        self.departure_time = 0
        self.travel_time = 0

def import_data():
    nodes = [Node(0, 0, float('inf'), 0),]
    N = int(input())
    for i in range(1, N+1):
        e, l, d = map(int, input().split())
        nodes.append(Node(i, e, l, d))
    time_matrix = []
    for i in range(N+1):
        time_matrix.append(list(map(int,input().split())))
    return N, nodes, time_matrix

def import_data_from_file(filepath):
    with open(filepath, 'r') as f:
        nodes = [Node(0, 0, float('inf'), 0),]
        N = int(f.readline())
        for i in range(1, N+1):
            e, l, d = map(int, f.readline().split())
            nodes.append(Node(i, e, l, d))
        time_matrix = []
        for i in range(N+1):
            time_matrix.append(list(map(int,f.readline().split())))
    return N, nodes, time_matrix



class Solver:
    def __init__(self, N, nodes, time_matrix):
        self.N = N
        self.nodes: list[Node] = nodes
        self.time_matrix = time_matrix
        self.route = [self.nodes[0]]

    def Insert(self, x: Node):
        '''Obj func: sum(max{0, y[i]-l[i]}) + total travel time'''
        min_obj_value = float('inf')
        best_pos = -1
        n = len(self.route)
        '''[0, 1,..., i-1, i,..., n ] -> [0, 1,..., i-1, x, i,..., n ]'''
        for i in range(1, n+1):
            '''Calculate the total travel time from the depot to x'''
            cur_travel_time = self.route[i-1].travel_time + self.time_matrix[self.route[i-1].ID][x.ID]
            '''Calculate the arrival time at x'''
            cur_time = self.route[i-1].departure_time + self.time_matrix[self.route[i-1].ID][x.ID]

            cur_obj_val = max(0, cur_time-x.l) + cur_travel_time

            if cur_time < x.e:
                cur_time = x.e
            cur_time += x.d

            for j in range(i,n):
                if j==i:
                    '''Calculate the total travel time from x to j'''
                    cur_travel_time  += self.time_matrix[x.ID][self.route[j].ID]
                    '''Calculate the arrival time at j'''
                    cur_time += self.time_matrix[x.ID][self.route[j].ID]
                else:
                    '''Calculate the total travel time from j-1 to j'''
                    cur_travel_time  += self.time_matrix[self.route[j-1].ID][self.route[j].ID]
                    '''Calculate the arrival time at j'''
                    cur_time += self.time_matrix[self.route[j-1].ID][self.route[j].ID]

                cur_obj_val = cur_obj_val + max(0, cur_time-self.route[j].l) + cur_travel_time

                cur_time += self.route[j].d
            
            if cur_obj_val < min_obj_value:
                best_pos = i
                min_obj_value = cur_obj_val

        if best_pos != -1:
            self.route.insert(best_pos, x)
            self.calc_time()
    
    def calc_time(self):
        for i in range(1,len(self.route)):
            node = self.route[i]
            prev_node = self.route[i-1]
            cur_time = prev_node.departure_time + self.time_matrix[prev_node.ID][node.ID]
            travel_time = prev_node.travel_time + self.time_matrix[prev_node.ID][node.ID]
            cur_time = max(cur_time, node.e) + node.d
            node.travel_time = travel_time
            node.departure_time = cur_time

    def checkFeasible(self):
        if len(self.route) != len(self.nodes):
            return 'Not complete' 
        for i in range(1, len(self.route)):
            cur_time = self.route[i-1].departure_time + self.time_matrix[self.route[i-1].ID][self.route[i].ID]
            if cur_time < self.route[i].e:
                cur_time = self.route[i].e
            if cur_time > self.route[i].l:
                return 'Non-feasible'
        return 'Feasible'


    def Solve(self):
        nodes = self.nodes[1:]
        nodes.sort(key = lambda node: node.l)
        for node in nodes:
            self.Insert(node)


    def printSol(self, filepath=None):
        print(len(self.route)-1)
        route = [node.ID for node in self.route]
        print(*route[1:])
        # print(TSPTW_cost(read_input_file(filepath), route[1:]))

if __name__ == '__main__':
    folder_path = 'E:\\SchoolworkBK\\20231\\Optimization\\TSPTW\\input'
    file_names = os.listdir(folder_path)
    for filename in file_names:
        print(f'{filename}:', end=' ')
        inp_filepath = os.path.join(folder_path,filename)
        N, nodes, time_matrix = import_data_from_file(inp_filepath)
        # N, nodes, time_matrix = import_data()
        s = Solver(N, nodes, time_matrix)
        s.Solve()
        s.printSol()
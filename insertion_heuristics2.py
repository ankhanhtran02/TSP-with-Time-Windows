from test_generator import TSPTW_cost, read_input_file
import os
from sys import exit

class Node:
    def __init__(self, ID, e, l, d):
        self.ID = ID
        self.e = e
        self.l = l
        self.d = d
        self.departure_time = 0

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
        min_total_time = float('inf')
        best_pos = -1
        n = len(self.route)
        '''[0, 1,..., i-1, i,..., n ] -> [0, 1,..., i-1, x, i,..., n ]'''

        for i in range(1, n+1):
            '''Calculate the arrival time at x'''
            cur_time = self.route[i-1].departure_time + self.time_matrix[self.route[i-1].ID][x.ID]

            if cur_time < x.e:
                cur_time = x.e
            if cur_time > x.l:
                break
            cur_time += x.d

            endRoute = True
            for j in range(i,n):
                if j==i:
                    '''Calculate the arrival time at j'''
                    cur_time += self.time_matrix[x.ID][self.route[j].ID]
                else:
                    '''Calculate the arrival time at j'''
                    cur_time += self.time_matrix[self.route[j-1].ID][self.route[j].ID]
                
                if cur_time < self.route[j].e:
                    cur_time = x.e
                if cur_time > self.route[j].l:
                    endRoute = False
                    break
                cur_time += self.route[j].d
            
            if endRoute and cur_time < min_total_time:
                best_pos = i
                min_total_time = cur_time

        if best_pos != -1:
            self.route.insert(best_pos, x)
            self.calc_time()
    
    def calc_time(self):
        for i in range(1,len(self.route)):
            node = self.route[i]
            prev_node = self.route[i-1]
            cur_time = prev_node.departure_time + self.time_matrix[prev_node.ID][node.ID]
            if cur_time <= node.l:
                cur_time = max(cur_time, node.e) + node.d
            else:
                print('Not feasible')
                exit()
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


    def printSol(self, filepath):
        # print(len(self.route)-1)
        route = [node.ID for node in self.route]
        # print(*route[1:])
        # print(self.checkFeasible())
        print(TSPTW_cost(read_input_file(filepath), route[1:]))

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
        s.printSol(inp_filepath)
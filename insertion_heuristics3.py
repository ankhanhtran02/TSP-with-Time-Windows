from test_generator import TSPTW_cost, read_input_file
import os
from sys import exit
import time
import random
import pandas as pd

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
        self.opti_solution = None

    def Insert(self, x: Node):
        min_total_time = float('inf')
        best_pos = -1
        n = len(self.route)
        '''[0, 1,..., i-1, i,..., n ] -> [0, 1,..., i-1, x, i,..., n ]
        Consecutively add x into n spaces in the current route to see the best position for x
        '''
        for i in range(1, n+1):
            '''Calculate the arrival time at x. If arrive earlier than e, arrival time = e'''
            cur_time = self.route[i-1].departure_time + self.time_matrix[self.route[i-1].ID][x.ID]

            if cur_time < x.e:
                cur_time = x.e
            '''If arrival time at x exceeds time limit, immediately move to the next position'''
            if cur_time > x.l:
                continue
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
                '''If arrival time at a node exceeds time limit, immediately move to the next position'''
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
            return True
        else:
            return False

    def calc_time(self):
        # print([node.ID for node in self.route])
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

    def checkFeasible(self, route):

        if len(route) != len(self.nodes):
            return False
        for i in range(1, len(route)):
            cur_time = route[i-1].departure_time + self.time_matrix[route[i-1].ID][route[i].ID]
            if cur_time < route[i].e:
                cur_time = self.route[i].e
            if cur_time > route[i].l:
                return False
        return True

    def cost(self, route):
        if not self.checkFeasible(route):
            return None
        travel_time = 0
        for i in range(1, len(route)):
            travel_time += self.time_matrix[route[i-1].ID][route[i].ID]
        return travel_time

    def InsertProcess(self, order):
        '''route: the current route, nodes will be added into it'''
        self.route = [self.nodes[0],]
        for i in order:
            if not self.Insert(self.nodes[i]):
                break
        else:
            c = self.cost(self.route)
            if c!= None and c < self.min_obj_value:
                self.min_obj_value = c
                self.opti_solution = self.route
                self.runtime = time.time() - self.start

    def Solve(self, maxtime):
        self.start = time.time()
        self.min_obj_value = float('inf')
        self.runtime = 0
        n = self.N
        sorted_by_e = sorted(self.nodes[1:], key = lambda node: node.e)
        l = [node.ID for node in sorted_by_e]
        self.InsertProcess(l)
        sorted_by_l = sorted(self.nodes[1:], key = lambda node: node.l)
        l = [node.ID for node in sorted_by_l]
        self.InsertProcess(l)
        while time.time() - self.start <= maxtime:
            '''l: the order the nodes are added'''
            l = random.sample(list(range(1, n+1)),n)
            self.InsertProcess(l)

        if self.min_obj_value != float('inf'):
            route = [node.ID for node in self.opti_solution]
            return route[1:], self.min_obj_value, self.runtime
        else:
            return 'None', 'None', 'None'

    def printSol(self):
        if self.opti_solution == None:
            print('No feasible solution')
        else:
            print(len(self.opti_solution)-1)
            route = [node.ID for node in self.opti_solution]
            print(*route[1:])
            print(self.cost(self.opti_solution))

if __name__ == '__main__':
    N, nodes, time_matrix = import_data()
    s = Solver(N, nodes, time_matrix)
    path, value, runtime = s.Solve(180)
    print(N)
    print(*path)

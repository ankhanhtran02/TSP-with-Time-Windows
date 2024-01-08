from ortools.linear_solver import pywraplp
from test_generator import *
from urllib.request import urlopen
from timeit import default_timer

BIG_M = 9999999


def solve(problem_inputs):
    # Create the linear solver with the GLOP backend
    n, e, l, d, t = problem_inputs
    e = [None] + e
    l = [None] + l
    d = [0] + d
    solver = pywraplp.Solver.CreateSolver("SAT")
    solver.SetTimeLimit(180*1000)
    if not solver:
        return

    start_solving = default_timer()
    TSP_path = []

    # Create variables
    # x = [
    #     [solver.IntVar(0, 1 - int(i == j), f"x_{i}_{j}") for j in range(n + 1)]
    #     for i in range(n + 1)
    # ]  
    # x_i_j = 1 if we go from point i to point j, else it is 0
    x = []
    for i in range(n+1):
        temp = []
        for j in range(n+1):
            if j==i or (i >= 1 and j >= 1 and e[i] > l[j]):
                temp.append(solver.IntVar(0, 0, f"x_{i}_{j}"))
            else:
                temp.append(solver.IntVar(0, 1, f"x_{i}_{j}"))
        x.append(temp)
    times_passed = [solver.IntVar(0, 0, f"times_passed_0")]
    for i in range(1, n + 1):
        times_passed.append(solver.IntVar(e[i], l[i], f"times_passed_{i}"))
    # times_passed.append(solver.IntVar(0, solver.infinity(), f'times_passed{n+1}'))
    y = [None] + [solver.IntVar(0, 1, f"y_{i}") for i in range(1, n + 1)]

    # Adding constraints
    for i in range(n + 1):
        ct = solver.Constraint(1, 1, f"ct_row_{i}")
        for j in range(n + 1):
            ct.SetCoefficient(x[i][j], 1)
    for j in range(n + 1):
        ct = solver.Constraint(1, 1, f"ct_col_{j}")
        for i in range(n + 1):
            ct.SetCoefficient(x[i][j], 1)

    for i in range(n + 1):
        for j in range(1, n + 1):
            if j != i and not (i >= 1 and e[i] > l[j]):
                # ct1 = solver.Constraint(
                #     -solver.infinity(), BIG_M - e[j], f"ct_{i}_{j}_1"
                # )
                # ct1.SetCoefficient(x[i][j], BIG_M)
                # ct1.SetCoefficient(times_passed[j], -1)

                ct2 = solver.Constraint(
                    -solver.infinity(), BIG_M - d[i] - t[i][j], f"ct_{i}_{j}_2"
                )
                ct2.SetCoefficient(x[i][j], BIG_M)
                ct2.SetCoefficient(times_passed[j], -1)
                ct2.SetCoefficient(times_passed[i], 1)

                # ct3 = solver.Constraint(
                #     -solver.infinity(), 2 * BIG_M + e[j], f"ct_{i}_{j}_3"
                # )
                # ct3.SetCoefficient(x[i][j], BIG_M)
                # ct3.SetCoefficient(times_passed[j], 1)
                # ct3.SetCoefficient(y[j], BIG_M)

                # ct4 = solver.Constraint(
                #     -solver.infinity(), BIG_M + d[i] + t[i][j], f"ct_{i}_{j}_4"
                # )
                # ct4.SetCoefficient(x[i][j], BIG_M)
                # ct4.SetCoefficient(times_passed[j], 1)
                # ct4.SetCoefficient(times_passed[i], -1)
                # ct4.SetCoefficient(y[j], -BIG_M)

    # Set solver objective
    objective = solver.Objective()
    for i in range(n + 1):
        for j in range(n + 1):
            if j!=0:
                objective.SetCoefficient(x[i][j], t[i][j])
    objective.SetMinimization()
    # Solving
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        # print(f'ORTOOLS status is {status}')
        # print(objective.Value())
        # print(solver.wall_time())
        # for i in range(n+1):
        #     for j in range(n+1):
        #         print(int(x[i][j].solution_value()), end=' ')
        #     print()
        i = 0
        print(n)
        while True:
            for j in range(n + 1):
                if x[i][j].solution_value() == 1:
                    if j != 0:
                        print(j, end=" ")
                        TSP_path.append(j)
                    i = j
                    break
            if i == 0:
                print()
                break
    elif status == pywraplp.Solver.NOT_SOLVED:
        # print('BUG')
        i = 0
        print(n)
        while True:
            for j in range(n + 1):
                if x[i][j].solution_value() == 1:
                    if j != 0:
                        print(j, end=" ")
                        TSP_path.append(j)
                    i = j
                    break
            if i == 0:
                print()
                break
    else:
        print(status)
    end_solving = default_timer()
    # print(f"Execution time is: {end_solving - start_solving}")
    return TSP_path


if __name__ == "__main__":

    # print("Start solving...")
    filename = 'new_test_cases/B10.txt'
    n, e, l, d, t = read_input_file(filename)
    TSP_path = solve((n, e, l, d, t))
    if len(TSP_path) > 0:
        # print(f'The optimal cost is: {TSPTW_cost(read_input_file(filename),TSP_path)}')
    else:
        print(f'Cannot find any feasible solution in time limit.')



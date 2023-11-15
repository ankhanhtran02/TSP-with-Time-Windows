from ortools.linear_solver import pywraplp
BIG_M = 9999999

def solve():
    # Create the linear solver with the GLOP backend
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        return
    # Input for the program
    n = int(input())
    e, l, d = [None], [None], [0]
    for _ in range(1, n+1):
        e_i, l_i, d_i = map(int, input().split())
        e.append(e_i)
        l.append(l_i)
        d.append(d_i)
    t = []
    for _ in range(n+1):
        t.append(list(map(int, input().split())))
    
    # Create variables
    x = [[solver.IntVar(0, 1 - int(i==j), f'x_{i}_{j}') for j in range (n+1)] for i in range(n+1)] # x_i_j = 1 if we go from point i to point j, else it is 0
    times_passed = [solver.IntVar(0, 0, f'times_passed_0')]
    for i in range(1, n+1):
        times_passed.append(solver.IntVar(e[i], l[i], f'times_passed_{i}'))
    # times_passed.append(solver.IntVar(0, solver.infinity(), f'times_passed{n+1}'))
    y = [None] + [solver.IntVar(0, 1, f'y_{i}') for i in range(1, n+1)]

    # Adding constraints
    for i in range(n+1):
        ct = solver.Constraint(1, 1, f'ct_row_{i}')
        for j in range(n+1):
            ct.SetCoefficient(x[i][j], 1)
    for j in range(n+1):
        ct = solver.Constraint(1, 1, f'ct_col_{j}')
        for i in range(n+1):
            ct.SetCoefficient(x[i][j], 1)

    for i in range(n+1):
        for j in range(1, n+1):
            if j != i:
                ct1 = solver.Constraint(-solver.infinity(), BIG_M - e[j], f'ct_{i}_{j}_1')
                ct1.SetCoefficient(x[i][j], BIG_M)
                ct1.SetCoefficient(times_passed[j], -1)


                ct2 = solver.Constraint(-solver.infinity(), BIG_M - d[i] - t[i][j], f'ct_{i}_{j}_2')
                ct2.SetCoefficient(x[i][j], BIG_M)
                ct2.SetCoefficient(times_passed[j], -1)
                ct2.SetCoefficient(times_passed[i], 1)


                ct3 = solver.Constraint(-solver.infinity(), 2 * BIG_M + e[j], f'ct_{i}_{j}_3')
                ct3.SetCoefficient(x[i][j], BIG_M)
                ct3.SetCoefficient(times_passed[j], 1)
                ct3.SetCoefficient(y[j], BIG_M)


                ct4 = solver.Constraint(-solver.infinity(), BIG_M + d[i] + t[i][j], f'ct_{i}_{j}_4')
                ct4.SetCoefficient(x[i][j], BIG_M)
                ct4.SetCoefficient(times_passed[j], 1)
                ct4.SetCoefficient(times_passed[i], -1)
                ct4.SetCoefficient(y[j], -BIG_M)
            


    # Set solver objective
    objective = solver.Objective()
    for i in range(n+1):
        for j in range(n+1):
            objective.SetCoefficient(x[i][j], t[i][j])
    objective.SetMinimization()
    # Solving
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        # print(objective.Value())
        # print(solver.wall_time())
        # for i in range(n+1):
        #     for j in range(n+1):
        #         print(int(x[i][j].solution_value()), end=' ')
        #     print()
        i = 0
        print(n)
        while True:
            for j in range(n+1):
                if x[i][j].solution_value() == 1:
                    if j != 0:
                        print(j, end=' ')
                    i = j
                    break
            if i == 0:
                print()
                break
        
    else:
        print(status)

if __name__ == "__main__":
    solve()


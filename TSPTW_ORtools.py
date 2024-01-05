from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def create_data_model():
    """Stores the data for the problem."""
    N = int(input())
    data = {}
    data["time_windows"] = [(0,0),]
    d_list=[0,]
    for _ in range(N):
        e,l,d = map(int, input().split())
        data["time_windows"].append((e,l))
        d_list.append(d)
    data["time_matrix"] = []
    for i in range(N+1):
        line = list(map(int,input().split()))
        new_line = []
        for j in range(N+1):
            if j==i:
                new_line.append(0)
            else:
                new_line.append(line[j]+d_list[i])
        data["time_matrix"].append(new_line)
    data["num_vehicles"] = 1
    data["depot"] = 0
    return N,data

def print_solution(N,data, manager, routing, solution):
    """Prints solution on console."""
    print(N)
    # print(f"Objective: {solution.ObjectiveValue()}")
    # time_dimension = routing.GetDimensionOrDie("Time")
    # total_time = 0
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = ""
        while not routing.IsEnd(index):
            # time_var = time_dimension.CumulVar(index)
            if index !=0:
                plan_output += (
                    f"{manager.IndexToNode(index)} "
                )
            # plan_output += (                     
            # f" Time({solution.Min(time_var)},{solution.Max(time_var)})"
            #         " -> "
            # )
            index = solution.Value(routing.NextVar(index))
        # time_var = time_dimension.CumulVar(index)
        # plan_output += (
        #     f"{manager.IndexToNode(index)}"
            # f" Time({solution.Min(time_var)},{solution.Max(time_var)})\n"
        # )
        # plan_output += f"Time of the route: {solution.Min(time_var)}min\n"
        print(plan_output)
        # total_time += solution.Min(time_var)
    # print(f"Total time of all routes: {total_time}min")


def main():
    """Solve the VRP with time windows."""
    # Instantiate the data problem.
    N, data = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        N+1, data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["time_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Time Windows constraint.
    time = "Time"
    routing.AddDimension(
        transit_callback_index,
        999999,  # allow waiting time
        999999999999,  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        time,
    )
    time_dimension = routing.GetDimensionOrDie(time)
    # Add time window constraints for each location except depot.
    for location_idx, time_window in enumerate(data["time_windows"]):
        if location_idx == data["depot"]:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
    # Add time window constraints for each vehicle start node.
    depot_idx = data["depot"]
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(
            data["time_windows"][depot_idx][0], data["time_windows"][depot_idx][1]
        )

    # Instantiate route start and end times to produce feasible times.
    for i in range(data["num_vehicles"]):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i))
        )
        routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(i)))

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(N,data, manager, routing, solution)


if __name__ == "__main__":
    main()


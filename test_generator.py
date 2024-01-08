import random
import numpy as np

def generate_testfile_X(time_window_range:tuple, travel_time_range:tuple, N=100, dist_type='uniform', filename='TSPTW_test_1.txt'):
    '''
    This function generates test cases for the TSP with Time Window problem. The parameters are:
        + time_window_range: a tuple (start, end) specifying the range of the time windows (e_i, l_i) (guarantee that e_i >= start and l_i ~ end)
        + travel_time_range: a tuple (start, end) specifying the range of the elements of the travel time matrix. Currently this travel time matrix is a (N+1) x (N+1) symmetric matrix with 0s on the main diagonal
        + N: the number of total points to travel
        + dist_type is a string which specifies the type of distribution you want to use when elements of the time window and travel time matrices are generated randomly
        + filename is the name of the file that stores the generated test case
        + This function generates test cases by generating a path first and then expand from it, so it will always guarantee a solution. It returns that path as a feasible solution for reference
    '''
    assert (time_window_range[0] == 0) and (time_window_range[1] > time_window_range[0]) and (travel_time_range[1] > travel_time_range[0]) and (N > 0)
    if (time_window_range[1] - time_window_range[0]) // (N+1) < sum(travel_time_range)//2:
        print('Conditions not suitable for generating test cases. Try shrinking the travel_time_range, expand the time_window_range, or decrease the value of N')
        return None
    file_handle = open(filename, 'w+')
    
    path = np.concatenate((np.array([0]), np.random.permutation(N) + 1))
    times_1 = []

    if dist_type == 'normal':
        temp = np.clip(np.random.normal(0.5, 0.1, N+1), 0.3, 0.7)
        temp = temp/np.sum(temp)
        temp = temp[:-1]
        times = [0]
        for i in range(N):
            times.append(int(times[i] + temp[i]*time_window_range[1]))
        times_1 = np.array(times).astype(int)
    elif dist_type == 'uniform':
        temp = np.random.uniform(0.3, 0.7, size=(N+1))
        temp = temp/np.sum(temp)
        temp = temp[:-1]
        times = [0]
        for i in range(N):
            times.append(int(times[i] + temp[i]*time_window_range[1]))
        times_1 = np.array(times).astype(int)
    else:
        file_handle.close()
        print('Unknown distribution type for generator')
        return None
    # times_1 = np.concatenate((np.array([0]), np.sort(np.random.randint(time_window_range[0], time_window_range[1], N))))
    times_2 = np.concatenate((times_1[1:], np.array([time_window_range[1]])))
    delta_times = times_2 - times_1
    # while np.any(delta_times < travel_time_range[0] + 10):
    #     times_1 = np.concatenate((np.array([0]), np.sort(np.random.randint(time_window_range[0], time_window_range[1], N))))
    #     times_2 = np.concatenate((times_1[1:], np.array([time_window_range[1]])))
    #     delta_times = times_2 - times_1
    travel_times_sample = np.random.randint(travel_time_range[0], delta_times)
    service_times = delta_times - travel_times_sample
    service_times[0] = 0
    # print(service_times)
    e = np.zeros((N+1, )).astype(int)
    l = np.zeros((N+1, )).astype(int)
    d = np.zeros((N+1, )).astype(int)
    for i in range(1, N+1):
        e[path[i]] = times_1[i]
        l[path[i]] = random.randint(e[path[i]], time_window_range[1])
        d[path[i]] = service_times[i]
    # print(travel_times_sample)
    travel_times = np.zeros((N+1, N+1))
    for i in range(N):
        travel_times[path[i]][path[i+1]] = travel_times_sample[i]
        travel_times[path[i+1]][path[i]] = travel_times_sample[i]
    travel_times[path[N]][0] = travel_times_sample[N]
    travel_times[0][path[N]] = travel_times_sample[N]
    for i in range(N+1):
        for j in range(i+1, N+1):
            if travel_times[i][j]==0:
                travel_times[i][j] = random.randint(travel_time_range[0], travel_time_range[1])
                travel_times[j][i] = travel_times[i][j]
    # print(travel_times)

    # Write things to the file
    file_handle.write(str(N)+'\n')
    for i in range(1, N+1):
        file_handle.write(f'{e[i]} {l[i]} {d[i]}\n')
    np.savetxt(file_handle, travel_times, fmt='%d', delimiter=' ')

    file_handle.close()
    return path[1:]

def convert_no_D_inputs(problem_inputs:tuple):
    N, e, l, d, t = problem_inputs
    for i in range(len(d)):
        for j in range(N+1):
            t[i+1][j] += d[i]
    d = [0 for _ in range(len(d))]
    return (N, e, l, d, t)

def generate_testfile_Y(time_window_range:tuple, travel_time_range:tuple, N=100, dist_type='uniform', filename='TSPTW_test_1.txt'):
    '''
    This function generates test cases for the TSP with Time Window problem. The parameters are:
        + time_window_range: a tuple (start, end) specifying the range of the time windows (e_i, l_i) (guarantee that e_i >= start and l_i ~ end)
        + travel_time_range: a tuple (start, end) specifying the range of the elements of the travel time matrix. This is an updated version of generate_testfile_X where all values of d are zeros and the time matrix will change accordingly, transform the problem into an equivalent one.
        + N: the number of total points to travel
        + dist_type is a string which specifies the type of distribution you want to use when elements of the time window and travel time matrices are generated randomly
        + filename is the name of the file that stores the generated test case
        + This function generates test cases by generating a path first and then expand from it, so it will always guarantee a solution. It returns that path as a feasible solution for reference
    '''
    assert (time_window_range[0] == 0) and (time_window_range[1] > time_window_range[0]) and (travel_time_range[1] > travel_time_range[0]) and (N > 0)
    if (time_window_range[1] - time_window_range[0]) // (N+1) < sum(travel_time_range)//2:
        print('Conditions not suitable for generating test cases. Try shrinking the travel_time_range, expand the time_window_range, or decrease the value of N')
        return None
    file_handle = open(filename, 'w+')
    
    path = np.concatenate((np.array([0]), np.random.permutation(N) + 1))
    times_1 = []

    if dist_type == 'normal':
        temp = np.clip(np.random.normal(0.5, 0.1, N+1), 0.3, 0.7)
        temp = temp/np.sum(temp)
        temp = temp[:-1]
        times = [0]
        for i in range(N):
            times.append(int(times[i] + temp[i]*time_window_range[1]))
        times_1 = np.array(times).astype(int)
    elif dist_type == 'uniform':
        temp = np.random.uniform(0.3, 0.7, size=(N+1))
        temp = temp/np.sum(temp)
        temp = temp[:-1]
        times = [0]
        for i in range(N):
            times.append(int(times[i] + temp[i]*time_window_range[1]))
        times_1 = np.array(times).astype(int)
    else:
        file_handle.close()
        print('Unknown distribution type for generator')
        return None
    # times_1 = np.concatenate((np.array([0]), np.sort(np.random.randint(time_window_range[0], time_window_range[1], N))))
    times_2 = np.concatenate((times_1[1:], np.array([time_window_range[1]])))
    delta_times = times_2 - times_1
    # while np.any(delta_times < travel_time_range[0] + 10):
    #     times_1 = np.concatenate((np.array([0]), np.sort(np.random.randint(time_window_range[0], time_window_range[1], N))))
    #     times_2 = np.concatenate((times_1[1:], np.array([time_window_range[1]])))
    #     delta_times = times_2 - times_1
    travel_times_sample = np.random.randint(travel_time_range[0], delta_times)
    service_times = delta_times - travel_times_sample
    service_times[0] = 0
    # print(service_times)
    e = np.zeros((N+1, )).astype(int)
    l = np.zeros((N+1, )).astype(int)
    d = np.zeros((N+1, )).astype(int)
    for i in range(1, N+1):
        e[path[i]] = times_1[i]
        l[path[i]] = random.randint(e[path[i]], time_window_range[1])
        d[path[i]] = service_times[i]
    # print(travel_times_sample)
    travel_times = np.zeros((N+1, N+1))
    for i in range(N):
        travel_times[path[i]][path[i+1]] = travel_times_sample[i]
        travel_times[path[i+1]][path[i]] = travel_times_sample[i]
    travel_times[path[N]][0] = travel_times_sample[N]
    travel_times[0][path[N]] = travel_times_sample[N]
    for i in range(N+1):
        for j in range(i+1, N+1):
            if travel_times[i][j]==0:
                travel_times[i][j] = random.randint(travel_time_range[0], travel_time_range[1])
                travel_times[j][i] = travel_times[i][j]
    # print(travel_times)

    # Converting...
    N, e, l, d, travel_times = convert_no_D_inputs((N, e[1:], l[1:], d[1:], travel_times))

    # Write things to the file
    file_handle.write(str(N)+'\n')
    for i in range(N):
        file_handle.write(f'{e[i]} {l[i]} {d[i]}\n')
    np.savetxt(file_handle, travel_times, fmt='%d', delimiter=' ')

    file_handle.close()
    return path[1:]

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
    
def TSPTW_cost(input:tuple, path:list, return_to_0=False):
    '''
    This function checks if a path satisfy the constraint for the TSPTW problem.
    If the path satisfy all constraints, the function returns the total travel cost.
    Else it will return None.
    The input parameter is a tuple (N, e, l, d, t) where:
        + N is the number of points to visit
        + e and l contain the starts and ends of N time windows
        + d is the service times at N points
        + t is the travel time matrix, size (N+1) x (N+1)
    You can use the return value from the read_input_file function for this parameter
    path is a permutation from 1 to N, the path to be evaluated
    If return_to_0 is set to True, the cost calculated will include the cost to return to point 0 from the last point of the path
    '''
    N, e, l, d, t = input
    total_time = 0
    path = [0] + [item for item in path]
    e = [None] + e
    l = [None] + l
    d = [0] + d
    travel_cost = 0
    for i in range(N):
        total_time += d[path[i]] + t[path[i]][path[i+1]]
        total_time = max(total_time, e[path[i+1]])
        travel_cost += t[path[i]][path[i+1]]
        if total_time > l[path[i+1]]:
            return None
        
    if return_to_0:
        travel_cost += t[path[-1]][0]
    return travel_cost

if __name__ == '__main__':
    generate_testfile_Y((0, 400), (10, 40), 12, dist_type='uniform')
    

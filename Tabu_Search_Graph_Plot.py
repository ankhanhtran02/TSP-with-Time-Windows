'''
    Welcome to my Graph Plotting Code of Tabu Search
    The result after running our own test case are saved in Tabu_value_table.xlsx
    Here is a breif guide on how to use
    a. Selecting a Mode
        1. Depth Plot: Plotting experiment for depths 
        2. Vertex Plot: Plotting experiment for vertices
        3. All Plot: PLotting the best value found in 1st experiments vs 2nd experiments vs final code
        4. Hill climbing vs Tabu Plot: Plotting the difference of Tabu and Hill Climbing

    b. Choose test case range
        There are 20 test cases listed, pick a number i and j and the code will Plot the test case ranging from i to j
    For best visualization, please choose 1 - 4 test cases at a time
'''

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("Tabu_value_table.xlsx")

desired_order = ["B10.txt", "B10_2.txt", "B20.txt", "B20_2.txt", "B30.txt", "B30_2.txt", "B63.txt", "B63_2.txt", "B201.txt", "B201_2.txt", "B233.txt", "B233_2.txt", "B300.txt", "B300_2.txt", "B500.txt", "B500_2.txt", "B800.txt", "B800_2.txt", "B1000.txt", "B1000_2.txt"]

df['Test_Case'] = pd.Categorical(df['Test_Case'], categories=desired_order, ordered=True)

def Mode_choosing(Mode):
    print("PLEASE SELECT TEST_CASE RANGE")
    print("LIST OF TEST_CASE AVAILABLE: (choose ranging from i -> j)")
    for i in range (20):
        print(i + 1, ":", desired_order[i])
    print("i = ", end = '')
    a = int(input())
    print("j = ", end = '')
    b = int(input())
    if Mode == 1:
        plot_depth(a, b)
    elif Mode == 2:
        plot_vertex(a, b)
    elif Mode == 3:
        plot_all(a, b)
    elif Mode == 4:
        plot_difference(a, b)
    else:
        return

def plot_depth(i, j):
    filtered_df = df[df['Test_Case'].between(desired_order[i-1], desired_order[j-1])]

    depth_columns = ['Depth1', 'Depth2', 'Depth3', 'Depth4', 'Depth5']

    ax = filtered_df.set_index('Test_Case')[depth_columns].T.plot(kind='line', marker='o')
    ax.set_xticks(range(len(depth_columns)))
    ax.set_xticklabels(depth_columns, rotation=45, ha='right', rotation_mode='anchor', fontsize=8)
    plt.title(f'Depth Plot for Test Case {i} to {j}')
    plt.xlabel('Depth Levels')
    plt.ylabel('Depth Values')
    plt.show()

def plot_vertex(i, j):
    filtered_df = df[df['Test_Case'].between(desired_order[i-1], desired_order[j-1])]

    depth_columns = ['Vertexn/2', 'Vertexn/3', 'Vertexn/4', 'Vertexn/5']

    ax = filtered_df.set_index('Test_Case')[depth_columns].T.plot(kind='line', marker='o')
    ax.set_xticks(range(len(depth_columns)))
    ax.set_xticklabels(depth_columns, rotation=45, ha='right', rotation_mode='anchor', fontsize=8)
    plt.title(f'Vertex Plot for Test Case {i} to {j}')
    plt.xlabel('Vertex Levels')
    plt.ylabel('Vertex Values')
    plt.show()

def plot_all(i, j):
    filtered_df = df[df['Test_Case'].between(desired_order[i-1], desired_order[j-1])].copy()

    depth_columns = ['Depth1']
    vertex_columns = ['Vertexn/2', 'Vertexn/3', 'Vertexn/4', 'Vertexn/5']
    final_column = ['Final']

    filtered_df.loc[:, 'Hill Climbing'] = filtered_df[depth_columns].min(axis=1)
    filtered_df.loc[:, 'Experiment'] = filtered_df[vertex_columns].min(axis=1)
    filtered_df.loc[:, 'Final'] = filtered_df[final_column]

    min_columns = ['Hill Climbing', 'Experiment', 'Final']

    ax = filtered_df.set_index('Test_Case')[min_columns].T.plot(kind='line', marker='o')
    plt.title(f'Minimum Values Plot for Test Case {i} to {j}')
    plt.xlabel('Categories')
    plt.ylabel('Minimum Values')
    plt.show()

def plot_difference(i, j):
    filtered_df = df[df['Test_Case'].between(desired_order[i-1], desired_order[j-1])]

    depth_columns = ['Depth1', 'Final']

    ax = filtered_df.set_index('Test_Case')[depth_columns].T.plot(kind='line', marker='o')
    ax.set_xticks(range(len(depth_columns)))
    ax.set_xticklabels(depth_columns, rotation=45, ha='right', rotation_mode='anchor', fontsize=8)
    plt.title(f'Hill Climbing vs Tabu Search')
    #plt.xlabel('Depth Levels')
    plt.ylabel('Depth Values')
    plt.show()

print("PLEASE SELECT A MODE: \n\
1: Plot Graph for Depth \n\
2: Plot Graph for Vertex \n\
3: PLot Graph for All \n\
4: PLot Hill Climbing vs Tabu")
Mode = int(input())
Mode_choosing(Mode)

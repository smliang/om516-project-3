import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import numpy as np
import warnings

def ReadTasksAndCreateProjectNetwork():
    tasks = pd.read_csv('Tasks.csv', index_col=0)#Read the project structure (task numbers, descriptions and precedence relationships). Note that all 95 past projects have the same structure.
#    tasks['xLevel'] = [0,1,1,2,3,4,5,6,7,8,  9,10,10,9,11,11,12,13,14,15, 15.5,15,16,16,15,16,16,15.5,17,18, 19,20,21,22]
#    tasks['yOrder'] = [0,1,-1,0,0,0,0,0,0,0, 0,0,1,-1,0,-1,0,0,0,0,        1,-3,4,5,-1,0,-1,2,0,0,         0,0,0,0]
    positions = _assignPositions(tasks)
    precedenceRelationships = []
    for successor in tasks.index:
        splitPreds = str(tasks.loc[successor,'Predecessors']).split(',')
        for pred in splitPreds:
            if pred != 'nan':
                predecessor = int(pred)
                precedenceRelationships.append((predecessor,successor))
    G = nx.DiGraph()
    for task in positions.index:
        G.add_node(task,pos=(positions.loc[task]['xLevel'], positions.loc[task]['yOrder']))
    pos=nx.get_node_attributes(G,'pos')
    for pair in precedenceRelationships:
        G.add_edge(pair[0], pair[1])
    return tasks, precedenceRelationships, G

def _assignPositions(tasks):
    positions = pd.DataFrame(index = tasks.index)
    positions['xLevel'] = [0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,9.5, 9,10,10,11,12,13,14,14.5,14,15,15,14,15,15,14.5,16,17, 18,19,20,21]
    positions['yOrder'] = [0, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0,  1,-1,-1,-2, 0, 0, 0, 0,   1,-4, 4, 5,-1,-2,-1,   2, 0, 0,  0, 0, 0, 0]
    return positions

def ReadTeamsDurations(team):
    '''Reads durations of the team's project's 34 tasks 
    '''
    allDurations = pd.read_csv('PastDurations.csv', index_col=1)
    teamsDurations = allDurations[allDurations['Project ID']==team][['Duration']]
    return teamsDurations

def ConvertTasksDataFrameToDictionaryAndAddSuccessorsColumn(df):
    pj_column = 'Duration'

    #single_project_data = df.set_index('Task')
    single_project_data = df.fillna('-')

    single_project_data['pj'] = single_project_data[pj_column]

    data_dict = single_project_data[['Predecessors', 'pj']].to_dict(orient= 'index')

    # for each task represented by the dictionary keys
    for task in data_dict.keys():
    
        # for both the predecessors and successors
        for node_group in ['Predecessors']:
        
            # if there are no predecessors or successors
            if data_dict[task][node_group] == '-':
            
                # update to an empty list
                data_dict[task][node_group] = []
            
            # else if there is a single predecessor or successor
            elif len(str(data_dict[task][node_group]).split(',')) == 1:
            
                # update to a list with the single predecessor or successor
                data_dict[task][node_group] = [int(data_dict[task][node_group])]
        
            # else if there are more than one predecessors or successors
            else:
            
                # split on the commas to create a list with multiple items
                data_dict[task][node_group] = [int(task) for task in data_dict[task][node_group].split(',')]
            
    # for each task represented by the dictionary keys
    for task in data_dict.keys():
        data_dict[task]['Successors'] = []
    
        # if there are no predecessors or successors
        if data_dict[task]['Predecessors']:
            for predecessor in data_dict[task]['Predecessors']:
                if data_dict[predecessor]['Successors']:
                    data_dict[predecessor]['Successors'].append(task)
                else:
                    data_dict[predecessor]['Successors'] = [task]
                
    return data_dict


def determine_critical_path(data_frame, start_time = 0):
    '''
    This function compute the critical path and makespan for a specified project. The 
    function creates a dictionary object (data_dictionary) from the given data frame of the project. The dictionary keys should
    be the names of the tasks, formatted as strings (even if the names are numeric). The 
    value for each key should be another dictionary with keys 'pj', 'Predecessors', and 
    and 'Successors'. The 'pj' value should be a number representing the task duration.
    The 'Predecessors' value should be a python list specifying the tasks that precede the
    associated task (formatted as strings). The 'Successors' value should be a python list 
    specifying the tasks that follow the associated task (formatted as strings). An empty list
    should be provided for the 'Predecessors'('Successors') value for all tasks that have no 
    predecessors(successors). An example of a valid dictionary object follows:
    
    {'1': {'pj': 4, 'Predecessors': [], 'Successors': ['4']},
     '2': {'pj': 6, 'Predecessors': [], 'Successors': ['5']},
     '3': {'pj': 10, 'Predecessors': [], 'Successors': ['6', '7']},
     '4': {'pj': 12, 'Predecessors': ['1'], 'Successors': ['6', '7']},
     '5': {'pj': 10, 'Predecessors': ['2'], 'Successors': ['6']},
     '6': {'pj': 2, 'Predecessors': ['3', '4', '5'], 'Successors': ['8']},
     '7': {'pj': 4, 'Predecessors': ['3', '4'], 'Successors': ['8']},
     '8': {'pj': 2, 'Predecessors': ['6', '7'], 'Successors': []}}
     
     The optional start_time argument specifies the time that the project starts.
     The optional include_plot argument should be set to True or False to indicate
     whether or not a plot of the project should be drawn. If a plot is drawn, tasks
     on the critical path will be colored red. Non-critical tasks will be colored green.   
     
    
    '''
    
    data_dictionary = ConvertTasksDataFrameToDictionaryAndAddSuccessorsColumn(data_frame)
    
    FP_Remaining_Tasks = list(data_dictionary.keys())

    tasks_to_remove = []
    for task in FP_Remaining_Tasks:
        if data_dictionary[task]['Predecessors'] == []:
            data_dictionary[task]['ES'] = start_time
            data_dictionary[task]['EF'] = start_time + data_dictionary[task]['pj']
            tasks_to_remove.append(task)
        
    FP_Remaining_Tasks = list(set(FP_Remaining_Tasks) - set(tasks_to_remove))
            
    while len(FP_Remaining_Tasks) > 0:
        for current_task in FP_Remaining_Tasks:
            if len(set(FP_Remaining_Tasks).intersection(set(data_dictionary[current_task]['Predecessors']))) == 0:
                FP_Remaining_Tasks.remove(current_task)
                break

        predecessors = data_dictionary[current_task]['Predecessors']
        max_ef = 0
        for predecessor in predecessors:
            if (data_dictionary[predecessor]['EF'] > max_ef):
                max_ef = data_dictionary[predecessor]['EF']
        data_dictionary[current_task]['ES'] = max_ef
        data_dictionary[current_task]['EF'] = data_dictionary[current_task]['ES'] + data_dictionary[current_task]['pj']
        
    BP_Remaining_Tasks = list(data_dictionary.keys())
    max_ef = 0
    for task in BP_Remaining_Tasks:
        if data_dictionary[task]['EF'] > max_ef:
            max_ef = data_dictionary[task]['EF']

    tasks_to_remove = []
    for task in BP_Remaining_Tasks:
        if data_dictionary[task]['Successors'] == []:
            data_dictionary[task]['LF'] = max_ef
            data_dictionary[task]['LS'] = max_ef - data_dictionary[task]['pj']
            tasks_to_remove.append(task)
        
    BP_Remaining_Tasks = list(set(BP_Remaining_Tasks) - set(tasks_to_remove))
            
    while len(BP_Remaining_Tasks) > 0:
        for current_task in BP_Remaining_Tasks:
            if len(set(BP_Remaining_Tasks).intersection(set(data_dictionary[current_task]['Successors']))) == 0:
                BP_Remaining_Tasks.remove(current_task)
                break

        successors = data_dictionary[current_task]['Successors']
        min_ls = np.inf
        for successor in successors:
            if (data_dictionary[successor]['LS'] < min_ls):
                min_ls = data_dictionary[successor]['LS']
        data_dictionary[current_task]['LF'] = min_ls
        data_dictionary[current_task]['LS'] = data_dictionary[current_task]['LF'] - data_dictionary[current_task]['pj']
        
    critical_path = []
    for key in data_dictionary:
        if np.round(data_dictionary[key]['ES'], 2) == np.round(data_dictionary[key]['LS'], 2):
            critical_path.append(key)
    
    return critical_path, max_ef, data_dictionary

def DrawGanttChart(data_dict, critical_nodes):
    import matplotlib.pyplot as plt

    schedule_dict = {}
    for task in data_dict.keys():
        schedule_dict[task] = {'start': data_dict[task]['ES'],
                            'end': data_dict[task]['ES'] + data_dict[task]['pj']}
    
    # Create a figure with a single subplot 
    fig, ax = plt.subplots(1,1,figsize = (12, 8)) 

    # Setting labels for x-axis and y-axis 
    ax.set_xlabel('Time') 
    ax.set_ylabel('Task') 

    # Setting labels for y-ticks 
    ax.set_yticks(range(len(schedule_dict.keys()))) 
    ax.set_yticklabels(list(schedule_dict.keys())) 

    # Setting figure title
    ax.set_title('Project Schedule')

    # Setting grid attribute 
    ax.grid(True) 

    # Adding bars for tasks
    for index, task in enumerate(schedule_dict.keys()):
        # Declaring a bar in schedule 
        if task in critical_nodes:
            ax.broken_barh([(schedule_dict[task]['start'], 
                             schedule_dict[task]['end'] - schedule_dict[task]['start'])],
                           (index-0.4, 0.8), 
                           edgecolor = 'k', 
                           color = 'r')
        else:
            ax.broken_barh([(schedule_dict[task]['start'], 
                             schedule_dict[task]['end'] - schedule_dict[task]['start'])],
                           (index-0.4, 0.8), 
                           edgecolor = 'k', 
                           color = 'g')

    # Invert the y-axis
    plt.gca().invert_yaxis()
    
    return fig
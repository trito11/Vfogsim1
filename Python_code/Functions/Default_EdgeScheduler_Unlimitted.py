import numpy as np
"""% TYPE: UNLIMITTED EDGE RESOURCE USAGE
% MULTI-TENANCY: OFF
% DETAILS: ALL THE USERS WITH SPECTRAL RESOURCES ARE ADMITTED TO EXECUTE
% THEIR TASKS. THE KEY ASSUMPTION IS THE NECESSARY MACHINES ARE ACTIVE. THE
% BASE STATION HAS UNLIMITTED EDGE RESOURCES SO THERE IS NO UPPER LIMIT OR
% OPTIMIZATION PROBLEM"""

def Default_EdgeScheduler_Unlimitted(UserList):
    Resource_Demand_loc = UserList.shape[1] - 4
    Is_Block_loc = UserList.shape[1] - 1
    Remaining_Time_loc = UserList.shape[1] - 2
    Spectral_Resource_loc = UserList.shape[1] - 7

    Total_Resources = np.sum(UserList[:, Resource_Demand_loc])
    UserList[:, Is_Block_loc] = 1

    for k in range(UserList.shape[0]):
        if UserList[k, Spectral_Resource_loc] > 0:
            UserList[:, Is_Block_loc] = 0  # The task is activated
            UserList[:, Remaining_Time_loc] = UserList[:, Remaining_Time_loc] - 1  # Task is executed

    return Total_Resources, UserList

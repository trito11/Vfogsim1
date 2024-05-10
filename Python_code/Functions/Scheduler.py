import numpy as np
from Default_Scheduler import Default_Scheduler
from Default_EdgeScheduler_Limitted import Default_EdgeScheduler_Limitted
from memory_profiler import profile
import gc

# @profile
def Scheduler(UserMatrix, CellMatrix, CellID, price_list, CellID_loc, Spectral_Resource_loc, GAMMA, Resource_Demand_loc, Is_Block_loc, Remaining_Time_loc, Demand_ServiceType_loc, bus, app_type, ServiceRequirements, ServiceUtilities, W, B_pow):
    count = 0
    upper_speed = 50

    if not np.any(UserMatrix[:, CellID_loc] == CellID):
        return UserMatrix, CellMatrix

    UserIndices = np.where(UserMatrix[:, CellID_loc] == CellID)[0]
    UserList = UserMatrix[UserIndices]
    Info_User = UserIndices + 1

    AppTypeIndices = UserList[:, app_type] 
    AppTypeIndices = AppTypeIndices.astype(int)

    R1 = ServiceRequirements[AppTypeIndices, 0]
    R2 = ServiceRequirements[AppTypeIndices, 1]
    R3 = ServiceRequirements[AppTypeIndices, 2]
    U1 = ServiceUtilities[AppTypeIndices, 0]
    U2 = ServiceUtilities[AppTypeIndices, 1]
    U3 = ServiceUtilities[AppTypeIndices, 2]
    Window_size = W[AppTypeIndices]

    SpectralResources = Default_Scheduler(np.log2(1 + UserList[:, 0] / GAMMA), upper_speed, R1, R2, R3, U1, U2, U3, Window_size)
    SpectralResources = np.around(SpectralResources, decimals=4)

    for k in range(Info_User.shape[0]): # Updating the user matrix
       UserMatrix[Info_User[k]-1, Spectral_Resource_loc] = SpectralResources[k]
       UserList[k][Spectral_Resource_loc] = SpectralResources[k]
       UserMatrix[Info_User[k]-1, app_type] = UserList[k][app_type]

    UserList = Default_EdgeScheduler_Limitted(B_pow, R3, R2, GAMMA, UserList, CellMatrix, price_list, Spectral_Resource_loc, Resource_Demand_loc, Is_Block_loc, Remaining_Time_loc, Demand_ServiceType_loc, bus)
    for k in range(UserList.shape[0]):
        UserMatrix[Info_User[k]-1, Is_Block_loc] = UserList[k, Is_Block_loc]
    CellMatrix[0] = np.sum(UserList[:, Is_Block_loc] * UserList[:, Resource_Demand_loc])
    

    return UserMatrix, CellMatrix

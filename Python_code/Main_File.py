import numpy as np
import pandas as pd
import sys
import os
import time
from pathlib import Path
link=Path(os.path.abspath(__file__))
link=link.parent
link=os.path.join(link, "Functions")
sys.path.append(link)
from preprocess import preprocess
from UserMatrix_Func import UserMatrix_Func
from Ozgur_User_driven_event import Ozgur_User_driven_event
from Scheduler import Scheduler
from Default_CellUpdate import Default_CellUpdate
 # assuming preprocess is a custom function or module
from memory_profiler import profile
import torch
from numba import jit, cuda 


source = "100"  # Dataset
EventRate = 0.8  # Less is more! varies between 0.5 =>1.1
servicediff = 0
traffic = preprocess("Input.csv", source)
traffic = np.array(traffic)
traffic = np.delete(traffic, 0, 1)
traffic = np.delete(traffic, 0, 0)
traffic[:, 1] += 1

bus = preprocess("bus.csv", source)
bus = np.array(bus)
bus = np.delete(bus, 0, 1)
bus = np.delete(bus, [3, 4, 5, 6], 1)
bus[:, :2] += 1

# Duration of simulation & users -- upperbounds
TimeFilter = 200
UserFilter = 500
Simulation_Duration = traffic[-1, 0]
USERCOUNT = max(traffic[:, 1])
CELLCOUNT = max(traffic[:, -1])
USERCOUNT = min(USERCOUNT, UserFilter)

def main():
    Simulation_Duration = min(Simulation_Duration, TimeFilter)

# SIMULATION PARAMETERS

    GAMMA = 0.000001
    price_list = np.array([1, 2, 5, 10])  # Price of certain services
    APP0_R = np.array([0.05, 0.07, 0.071])  # R1 R2 R3 %BG
    APP1_R = np.array([0.01, 0.075, 0.4])  # R1 R2 R3 %M2M
    APP2_R = np.array([0.1, 0.225, 0.55])  # R1 R2 R3 %Inelastic
    APP3_R = np.array([0, 1.083, 20])  # R1 R2 R3 %Elastic
    APP0_U = np.array([0, 1, 1])  # U1 U2 U3
    APP1_U = np.array([-1, 0.7, 1])  # U1 U2 U3
    APP2_U = np.array([-0.5, 0.7, 1])  # U1 U2 U3
    APP3_U = np.array([0, 1, 1.8])  # U1 U2 U3
    W = np.array([1, 1, 1, 1])  # Per App type

    Ttr = 1
    Texec = 1
    Tmig = 1
    # AUXILIARY VARIABLES
    SINR_loc = 1
    SpectralResources_loc = 2
    Demand_Time_loc = 3
    Demand_Resource_loc = 4
    Demand_ServiceType_loc = 5
    Cell_ID_loc = 6
    IsBlocked_loc = 7
    Rem_Time_loc = 8
    X_loc = 9
    Y_loc = 10
    Speed_loc = 11
    Completed_loc = 12
    Queue_Delay_loc = 13
    Cell_Change_loc = 14
    CELLCOUNT=int(CELLCOUNT)
    USERCOUNT=int(USERCOUNT)
    CellMatrix = np.zeros((CELLCOUNT, 3, Simulation_Duration),dtype=float)
    a=time.time()
    B_pow = 5000  # if it is 0 - no fog devices on buses
    UserMatrix = UserMatrix_Func(USERCOUNT, Simulation_Duration, traffic, SINR_loc, Cell_ID_loc, X_loc, Y_loc, Speed_loc, 14)
    ServiceRequirements = np.array([APP0_R, APP1_R, APP2_R, APP3_R])
    ServiceUtilities = np.array([APP0_U, APP1_U, APP2_U, APP3_U])
    # THE EVENT GENERATION ~ PER TTI
    events = np.zeros((1, 4))
    for n in range(1, Simulation_Duration + 1):
        port = UserMatrix[:, Cell_ID_loc-1, n-1] > 0
        dummy = Ozgur_User_driven_event(np.sum(port, 0), 3, 1, 1, EventRate, UserMatrix[:, Cell_ID_loc-1, n-1] > 0)
        dummy = np.transpose(dummy)
        dummy = dummy[~np.any(dummy == 0, axis=1)]
        dummy[:, 0] = n
        events = np.vstack((events, dummy))
    events = np.delete(events, 0, 0)
    for i in range(events.shape[0]):
        time_slot = int(events[i, 0])
        user = int(events[i, 1])
        app_type = events[i, 2]
        demand = events[i, 3]
        UserMatrix[user-1, Demand_ServiceType_loc-1, time_slot-1] = app_type
        UserMatrix[user-1, Demand_Time_loc-1, time_slot-1] = 1
        UserMatrix[user-1, Demand_Resource_loc-1, time_slot-1] = demand

    UserMatrix[:, Rem_Time_loc-1, 0] = UserMatrix[:, Demand_Time_loc-1, 0]
    # print(UserMatrix[0,:,0])
    Unserved = np.zeros(USERCOUNT)
    # SIMULATOR
    
    for n in range(1, Simulation_Duration + 1):
        for c in range(1, CELLCOUNT + 1):
            UserMatrix[:, :, n-1], CellMatrix[c - 1, :, n - 1] = Scheduler(UserMatrix[:, :, n-1], CellMatrix[c - 1, :, n - 1], c, price_list, Cell_ID_loc-1, SpectralResources_loc-1, GAMMA, Demand_Resource_loc-1, IsBlocked_loc-1, Rem_Time_loc-1, Demand_ServiceType_loc-1, bus[(bus[:, 0] == n) & (bus[:, 4] == c), :], Demand_ServiceType_loc-1, ServiceRequirements, ServiceUtilities, W, B_pow)
        if n > 1:
            CellMatrix[:, :, n - 1], UserMatrix[:, :, n-1] = Default_CellUpdate(CellMatrix[:, :, n - 2], UserMatrix[:, :, n - 2], UserMatrix[:, :, n-1], Cell_ID_loc-1, Demand_Resource_loc-1, IsBlocked_loc-1, Completed_loc-1, Rem_Time_loc-1, Cell_Change_loc-1)
        print(time.time()-a)
        for user in range(1, USERCOUNT + 1):

            y= UserMatrix[user - 1, Rem_Time_loc-1, n-1]
            if n > 1:
                UserMatrix[user - 1, Queue_Delay_loc-1, n-1] = UserMatrix[user - 1, Queue_Delay_loc-1, n - 2]
            if y > 0 and UserMatrix[user - 1, IsBlocked_loc-1, n-1] == 1 and UserMatrix[user - 1, SINR_loc-1, n-1] > 0:
                UserMatrix[user - 1, Queue_Delay_loc-1, n-1] += 1
            if n < Simulation_Duration:
                x= UserMatrix[user - 1, Demand_Time_loc-1, n ]
                if x > 0 and y > 0:
                    if n > 1:
                        Unserved[user - 1] += 1
                    UserMatrix[user - 1, Rem_Time_loc-1, n ] = UserMatrix[user - 1, Demand_Time_loc-1, n ]
                elif x == 0 and y >= 0:
                    UserMatrix[user - 1, Rem_Time_loc-1, n ] = UserMatrix[user - 1, Rem_Time_loc-1, n-1]
                else:
                    UserMatrix[user - 1, Rem_Time_loc-1, n] = UserMatrix[user - 1, Demand_Time_loc-1, n]
    print(time.time()-a)
    return UserMatrix, Unserved ,events , ServiceRequirements ,ServiceUtilities
 
if __name__ =="__main__":
    main()
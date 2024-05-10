import numpy as np
import os
import scipy.io as sio
from pathlib import Path
import sys
link=Path(os.path.abspath(__file__))
link=link.parent
link=os.path.join(link, "Functions")
sys.path.append(link)
from Acceptance_ratio import Acceptance_ratio
from ResourceDistribution import ResourceDistribution
from QoS_Cell_based import QoS_Cell_based
from plotting_general import plotting_general
from QoE_Achieved_rate import QoE_Achieved_rate
from Acceptance_ratio import Acceptance_ratio
from Acceptance_ratio import Acceptance_ratio
from QoS_Total_Delay import QoS_Total_Delay

# from Functions import Acceptance_ratio, ResourceDistribution, QoS_Cell_based, QoS_Cell_based, QoE_Achieved_rate, plotting_general
# from Main_File import  USERCOUNT, SINR_loc, SpectralResources_loc, UserMatrix, Unserved, events, IsBlocked_loc, Cell_ID_loc, Ttr, Tmig, Texec, Cell_Change_loc, Demand_Time_loc, Queue_Delay_loc, ServiceRequirements, ServiceUtilities
from Main_File import *
import numpy as np
import os

UPBOUND = 1
print(f"Will be run for {UPBOUND} instances")

for mircolo in range(1, UPBOUND + 1):
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"Step {mircolo} in progress")
    UserMatrix,Unserved,events,ServiceRequirements,ServiceUtilities=main()
    filename = rf"D:\Lab\Vfogsim\Python_code\results\{mircolo}.npz"
    variables_to_save = {'UserMatrix':UserMatrix, 'Unserved':Unserved,'events':events,'ServiceRequirements':ServiceRequirements,'ServiceUtilities':ServiceUtilities}
    np.savez(filename, **variables_to_save)

Mitico_Matrix = np.zeros((USERCOUNT,14,Simulation_Duration))
Mitico_AcceptanceRatio = np.zeros(USERCOUNT)
Mitico_RESDISP = np.zeros(4)
Mitico_Cell = np.zeros(12)

print("Completed")

# For Printing
for mitico in range(1, UPBOUND + 1):
    infile = f"{mitico}.npz"
    data = np.load(infile)
    globals().update(data)
    UserMatrix = data['UserMatrix']
    Unserved = data['Unserved']
    events = data['events']
    ServiceRequirements = data['ServiceRequirements']
    ServiceUtilities= data['ServiceUtilities']
    Mitico_Matrix += UserMatrix
    Mitico_AcceptanceRatio += Acceptance_ratio(Unserved, events)
    Mitico_RESDISP += ResourceDistribution(UserMatrix, events, IsBlocked_loc-1)
    Mitico_Cell += QoS_Cell_based(Cell_ID_loc-1, Ttr, Tmig, Texec, Cell_Change_loc-1, Demand_Time_loc-1, Queue_Delay_loc-1, UserMatrix)

Mitico_Matrix /= UPBOUND
Mitico_AcceptanceRatio /= UPBOUND
Mitico_RESDISP /= UPBOUND
Mitico_Cell /= UPBOUND

infile = f"{UPBOUND}.npz"
data = np.load(infile)
globals().update(data)


QoS_Total_Delay(Ttr, Tmig, Texec, Cell_Change_loc-1, Demand_Time_loc-1, Queue_Delay_loc-1, Mitico_Matrix)
QoE_Achieved_rate(np.log2(1 + Mitico_Matrix[:, SINR_loc, :] / GAMMA), Mitico_Matrix[:, SpectralResources_loc, :], ServiceRequirements, ServiceUtilities, Mitico_Matrix)
plotting_general(Mitico_Cell, 'Cell ID', 'Congestion Duration (%Time)')
plotting_general(Mitico_AcceptanceRatio, 'User ID', 'Request Acceptance Ratio')
plotting_general(Mitico_RESDISP.T, 'Service Type', 'Request Acceptance Ratio per Service Type')

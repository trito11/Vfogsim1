import numpy as np
import os
print(os.getcwd())
import sys
from pathlib import Path
link=Path(os.path.abspath(__file__))
link=link.parent
link=os.path.join(link, "Functions")
sys.path.append(link)
from TD_TD import TD_TD

def ResourceDistribution(UserMatrix, events, IsBlocked_loc):
    Total_Requests = np.zeros(4)
    Accepted_Requests = np.zeros(4)
    Hasta = TD_TD(UserMatrix[:,4,:], UserMatrix.shape[0], UserMatrix.shape[2])
    Blocked = TD_TD(UserMatrix[:,6,:], UserMatrix.shape[0], UserMatrix.shape[2])
    Res = TD_TD(UserMatrix[:,1,:], UserMatrix.shape[0], UserMatrix.shape[2])

    Total_Requests[0] = np.sum(Hasta==1)
    Total_Requests[1] = np.sum(Hasta==2)
    Total_Requests[2] = np.sum(Hasta==3)
    Total_Requests[3] = np.sum(Hasta==4)

    for i in range(Blocked.shape[0]):
        for j in range(Blocked.shape[1]):
            a=Hasta[i,j]
            if (Blocked[i,j]==0 and Hasta[i,j]>1):
                Accepted_Requests[int(Hasta[i,j])-1] += 1
            if(Hasta[i,j]==1 and Res[i,j]>0):
                Accepted_Requests[0] += 1

    if(np.sum(Hasta==1)==0): # No request is sent
        unconnected = np.sum(Hasta==0)
        Accepted_Requests[0] = Hasta.shape[0]-unconnected
        Total_Requests[0] = Hasta.shape[0]

    if(np.sum(Hasta==1)>0 and np.sum(Hasta==0)>0): # There is active requests but also unconnected
        unconnected = np.sum(Hasta==0)
        Total_Requests[0] += unconnected

    Total_Requests[Total_Requests==0] = 1
    T = Accepted_Requests*100/Total_Requests

    return T
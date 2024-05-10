import numpy as np
import sys
from pathlib import Path
import os
link=Path(os.path.abspath(__file__))
link=link.parent
link=os.path.join(link, "Functions")
sys.path.append(link)
from TD_TD import TD_TD

def QoS_Cell_based(Cell_ID_loc, Ttr, Tmig, Texec, Cell_Change_loc, Demand_Time_loc, Queue_Delay_loc, UserMatrix):
    dummy = TD_TD(UserMatrix[:, Cell_ID_loc, :], UserMatrix.shape[0], UserMatrix.shape[2])
    blocked = TD_TD(UserMatrix[:, Cell_ID_loc+1, :], UserMatrix.shape[0], UserMatrix.shape[2])
    dummy2 = TD_TD(UserMatrix[:, Queue_Delay_loc, :], UserMatrix.shape[0], UserMatrix.shape[2])
    type_ = TD_TD(UserMatrix[:, Demand_Time_loc+2, :], UserMatrix.shape[0], UserMatrix.shape[2])
    Cells = np.zeros(int(np.max(dummy)))
    Cell_trace = np.zeros((dummy.shape[1], int(np.max(dummy))))
    Cell_trace_users = np.zeros((dummy.shape[1], int(np.max(dummy))))
    Unconnected_users = blocked[~np.any(dummy, axis=1), :].shape[0]
    blocked = blocked[np.any(dummy, axis=1), :]
    dummy2 = dummy2[np.any(dummy, axis=1), :]
    type_ = type_[np.any(dummy, axis=1), :]
    dummy = dummy[np.any(dummy, axis=1), :]

    for i in range(dummy.shape[0]):
        for j in range(dummy.shape[1]):
            if dummy[i, j] == 0:
                blocked[i, j] = 1
            if dummy[i, j] > 0:
                Cell_trace_users[j, int(dummy[i, j])-1] += 1
                if blocked[i, j] > 0:
                    if Cell_trace[j, int(dummy[i, j])-1] == 0:
                        Cells[int(dummy[i, j])-1] += 1
                        Cell_trace[j, int(dummy[i, j])-1] = 1

    T = Cells * 100 / np.sum(Cell_trace_users != 0, axis=0)

    return T
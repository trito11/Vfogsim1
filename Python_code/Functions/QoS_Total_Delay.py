import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import os
link=Path(os.path.abspath(__file__))
link=link.parent
link=os.path.join(link, "Functions")
sys.path.append(link)
from TD_TD import TD_TD

def QoS_Total_Delay(Ttr, Tmig, Texec, Cell_Change_loc, Demand_Time_loc, Queue_Delay_loc, UserMatrix):
    # T queueing
    dummy = TD_TD(UserMatrix[:, Queue_Delay_loc, :], UserMatrix.shape[0], UserMatrix.shape[2])
    Tqueueing = dummy[:, -1]

    # T migration calculation
    pre_m = TD_TD(UserMatrix[:, Cell_Change_loc, :], UserMatrix.shape[0], UserMatrix.shape[2])
    m = np.max(pre_m, axis=1)
    Tmigration = Tmig * m

    # T execution calculation
    pre_r = TD_TD(UserMatrix[:, Demand_Time_loc, :], UserMatrix.shape[0], UserMatrix.shape[2])
    pre_r[pre_r < 0] = 0
    r = np.sum(pre_r, axis=1)
    Texecution = Texec * r

    # T transmission calculation
    Ttrans = r * 0.5 * Ttr

    # Plotting
    
    T={'Queueing Delay':Tqueueing,'Migration Delay':Tmigration}
    width = 0.5
    fig, ax = plt.subplots()
    bottom = np.zeros(100)
    for boolean, weight_count in T.items():
        p = ax.bar(np.arange(100), weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count
    # plt.bar(np.arange(T.shape[1]), T.T, stacked=True)

# Vẽ biểu đồ
    # plt.bar(np.arange(Tqueueing.shape[0]), [Tqueueing, Tmigration], stacked=True)

    plt.xlabel('Users ID')
    plt.ylabel('Total Delay')
    # plt.legend(['Queueing Delay', 'Migration Delay'])
    # plt.grid(True)
    plt.show()
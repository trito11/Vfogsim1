import numpy as np

def User_driven_event(N_user, N_appType, T_totalSim, flag, Event):
    ui_taskType = np.random.randint(N_appType, size=N_user) + 1
    appData = [[0.5e3, 250, .15], [1.0e3, 2500, .20], [1.5e3, 10000, .25]]
    N_cell = 1
    Epoch = .5
    Lambda = N_user/N_cell/Epoch
    Lambda *= Event
    t = -np.log(np.random.rand())/Lambda
    n = 0
    et = 0
    ui = []
    tt = []
    ts = []
    while t < T_totalSim:
        n += 1
        ui.append(np.random.randint(N_user) + 1)
        if flag == 0:
            tt.append(ui_taskType[ui[n-1]-1])
        else:
            tt.append(np.random.randint(N_appType) + 1)
        ts.append(appData[tt[n-1]-1][0] + appData[tt[n-1]-1][0]*(np.random.rand()*0.2 - 0.1))
        et = t
        t = t - np.log(np.random.rand())/Lambda
    eventInfo = [et, ui, tt, ts]
    return eventInfo
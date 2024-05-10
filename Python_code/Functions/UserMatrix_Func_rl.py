import numpy as np

def UserMatrix_Func(USERCOUNT, Simulation_Duration, traffic, SINR_loc, Cell_ID_loc, X_loc, Y_loc, Speed_loc, Num_Col):
    # Simulation_Duration = traffic[-1, 0]
    # USERCOUNT = np.max(traffic[:, 1])
    UserMatrix = np.zeros((USERCOUNT, Num_Col, Simulation_Duration))

    for l in range(traffic.shape[0]):
        time = int(traffic[l, 0])
        vehicle = int(traffic[l, 1])
        X = traffic[l, 5]
        Y = traffic[l, 6]
        SINR = traffic[l, traffic.shape[1] - 2]
        CellID = traffic[l, traffic.shape[1] - 1]
        speed = traffic[l, 4]   # Currently Not Active
        if time <= Simulation_Duration:
            if vehicle <= USERCOUNT:
                UserMatrix[vehicle, X_loc, time-1] = X
                UserMatrix[vehicle, Y_loc, time-1] = Y
                UserMatrix[vehicle, Speed_loc , time-1] = speed
                UserMatrix[vehicle, SINR_loc , time-1] = SINR
                UserMatrix[vehicle, Cell_ID_loc , time-1] = CellID

    return UserMatrix
import numpy as np

def UserMatrix_Func(USERCOUNT, Simulation_Duration, traffic, X_loc, Y_loc, sinr0, sinr1, sinr2, sinr3, sinr4, sinr5, sinr6, sinr7, sinr8, Num_Col):
    # Simulation_Duration = traffic[-1, 0]
    # USERCOUNT = np.max(traffic[:, 1])
    UserMatrix = np.zeros((USERCOUNT, Num_Col, Simulation_Duration))

    for l in range(traffic.shape[0]):
        time = int(traffic[l, 0])
        vehicle = int(traffic[l, 1])
        X = traffic[l, 2]
        Y = traffic[l, 3]
        

        # speed = traffic[l, 4]   # Currently Not Active
        if time <= Simulation_Duration:
            if vehicle <= USERCOUNT:
                UserMatrix[vehicle, X_loc, time-1] = X
                UserMatrix[vehicle, Y_loc, time-1] = Y
                UserMatrix[vehicle, sinr0, time-1] = traffic[l,4]
                UserMatrix[vehicle, sinr1, time-1] = traffic[l,5]
                UserMatrix[vehicle, sinr2, time-1] = traffic[l,6]
                UserMatrix[vehicle, sinr3, time-1] = traffic[l,7]
                UserMatrix[vehicle, sinr4, time-1] = traffic[l,8]
                UserMatrix[vehicle, sinr5, time-1] = traffic[l,9]
                UserMatrix[vehicle, sinr6, time-1] = traffic[l,10]
                UserMatrix[vehicle, sinr7, time-1] = traffic[l,11]
                UserMatrix[vehicle, sinr8, time-1] = traffic[l,12]

    return UserMatrix

def Cell_Init(CellMatrix, UserMatrix, CellID_loc, IsBlocked_loc, USERCOUNT, Demand_Resource_loc):
    # Assuming all the vehicles are always connected at to a base station
    UserMatrix[:, IsBlocked_loc] = 0
    for user in range(USERCOUNT):
        if UserMatrix[user, 0] == 0 and UserMatrix[user, CellID_loc] == 0:
            # Skip
            pass
        else:
            if CellMatrix[UserMatrix[user, CellID_loc], 0] + UserMatrix[user, CellID_loc] <= CellMatrix[UserMatrix[user, CellID_loc], 1]:
                CellMatrix[UserMatrix[user, CellID_loc], 0] += UserMatrix[user, Demand_Resource_loc]
            else:
                UserMatrix[user, IsBlocked_loc] = 1
    
    return UserMatrix, CellMatrix

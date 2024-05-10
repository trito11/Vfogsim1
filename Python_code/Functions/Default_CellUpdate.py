"""""
TYPE: UPDATE THE CELL DATA BASED ON THE USER INFORMATON
MULTI-TENANCY: OFF
DETAILS: UPDATES THE CELL PARAMETERS IN CASE: (1) USER MOVED TO ANOTHER
CELL, (2) THE TASK EXECUTION IS FINISHED
"""""

def Default_CellUpdate(CellMatrix, UserMatrix_Old, UserMatrix_New, CellID_loc, UserDemand_Location, Is_Blocked, IsCompleted, RemTime_loc, Cell_Change_loc):
    for user in range(UserMatrix_Old.shape[0]):
        # If the user moved to another cell
        UserMatrix_New[user, Cell_Change_loc] = UserMatrix_Old[user, Cell_Change_loc]
        if UserMatrix_Old[user, CellID_loc] != UserMatrix_New[user, CellID_loc]:
            UserMatrix_New[user, Cell_Change_loc] += 1  # Migrating the task

        # If the user is blocked
        if UserMatrix_Old[user, Is_Blocked] == 1:
            UserMatrix_New[user, RemTime_loc] = UserMatrix_Old[user, RemTime_loc]
        else:
            if UserMatrix_Old[user, CellID_loc] != 0 and UserMatrix_Old[user, UserDemand_Location] > 0:  # Is served
                dummy = UserMatrix_Old[user, RemTime_loc] - 1  # If the user is not blocked
                if dummy <= 0:
                    UserMatrix_New[user, IsCompleted] = UserMatrix_Old[user, IsCompleted] + 1
                    UserMatrix_New[user, RemTime_loc] = 0
                else:
                    UserMatrix_New[user, RemTime_loc] = dummy

    return CellMatrix, UserMatrix_New

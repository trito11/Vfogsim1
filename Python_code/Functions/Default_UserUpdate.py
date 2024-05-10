
def Default_UserUpdate(UserMatrix_old, UserMatrix_new):
    CellID_Location = len(UserMatrix_old[0]) - 4
    CellMatrix_Update = []
    counter = 0
    
    for user in range(len(UserMatrix_old)):
        if UserMatrix_old[user][CellID_Location] != UserMatrix_new[user][CellID_Location]:
            # Then the user has switched base station
            CellMatrix_Update.append([UserMatrix_old[user][CellID_Location], user, UserMatrix_new[user][CellID_Location]])
            counter += 1
    
    CellMatrix_Update.sort(key=lambda x: x[0])
    return CellMatrix_Update
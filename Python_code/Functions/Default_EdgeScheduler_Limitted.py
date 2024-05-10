
import numpy as np
from Default_UtilityFunction_Edge import *
import pulp
# import intlinprog_pulp
from intlinprog_gurobi import intlinprog_gurobi
from memory_profiler import profile
import gc


# @profile
def Default_EdgeScheduler_Limitted(B_pow, R3, R2, GAMMA, UserList_All, CellMatrix, price_list, Spectral_Resource_loc, Resource_Demand_loc, Is_Block_loc, Remaining_Time_loc, Demand_ServiceType_loc, bus):
    counter = 0
    UserList = []
    user_info = []

    # Lặp qua danh sách người dùng để kiểm tra có thể thực hiện tính toán vi mạch không
    for k in range(len(UserList_All)):
        UserList_All[k][Is_Block_loc] = 1
        if (np.log2(1 + UserList_All[k][0] / GAMMA) * UserList_All[k][Spectral_Resource_loc] >= 0 and UserList_All[k][Demand_ServiceType_loc] > 0):
            if (np.log2(1 + UserList_All[k][0] / GAMMA) * UserList_All[k][Spectral_Resource_loc] >= 1.15 * R2[k]):
                UserList.append(UserList_All[k])
                user_info.append(k)
                counter += 1

    # Nếu không có người dùng nào có thể thực hiện tính toán vi mạch, trả về ngay
    if counter == 0:
        return UserList_All

    USERCOUNT = len(UserList)
    UserList_All[:, Is_Block_loc] = 1

    # Hàm mục tiêu: Tối thiểu hóa tổng lượng tài nguyên được sử dụng cho mỗi người dùng
    c = np.concatenate((np.zeros(USERCOUNT), -np.ones(USERCOUNT)))
    OBJSIZE = len(c)

    # Xây dựng các thông số tiện ích
    UtilityPerUser = Default_UtilityFunction_Edge(UserList, price_list, OBJSIZE, USERCOUNT, Resource_Demand_loc, Remaining_Time_loc, Demand_ServiceType_loc)
    Val_UtilityFunction = [0] * USERCOUNT

    # Tổng tài nguyên được gán không vượt quá tài nguyên có sẵn
    Tot_AssignedResources = [UserList[i][Resource_Demand_loc] for i in range(USERCOUNT)] + [0] * (USERCOUNT)
    Val_AssignedResources = CellMatrix[1] + len(bus) * B_pow
    if Val_AssignedResources > 0:
        print(B_pow)

    # Xây dựng các ma trận ràng buộc
    B = [Tot_AssignedResources]+ UtilityPerUser
    B=np.array(B)
    b = [Val_AssignedResources]+Val_UtilityFunction
    b=np.array(b)
    A = None
    a = None

    # Giới hạn dưới và giới hạn trên cho các biến quyết định
    lb = [0] * (2 * USERCOUNT)
    ub = [1] * USERCOUNT + [np.inf] * USERCOUNT
    intcon = list(range(1, USERCOUNT + 1))

    # Giải bài toán tối ưu hóa bằng PuLP
    xsol  = intlinprog_gurobi(c, intcon, B, b, A, a, lb, ub)
    

    # Cập nhật UserList_All dựa trên kết quả
    res = xsol[:USERCOUNT]
    for user in range(USERCOUNT):
        if res[user] == 1:
            UserList_All[user_info[user]][Is_Block_loc] = 0
    gc.collect()

    return UserList_All

import math

def Default_UtilityFunction_Edge(UserList, price_list, OBJSIZE, USERCOUNT, Resource_Demand_loc, Remaining_Time_loc, Demand_ServiceType_loc):
    const = [[0] * OBJSIZE for _ in range(USERCOUNT)]
    
    for k in range(USERCOUNT):
        if math.log(UserList[k][Remaining_Time_loc ]) > 0:
            const[k][k] = -price_list[int(UserList[k][Demand_ServiceType_loc ])] * UserList[k][Resource_Demand_loc ] * 1 / math.log(UserList[k][Remaining_Time_loc ])
        else:
            const[k][k] = -price_list[int(UserList[k][Demand_ServiceType_loc ])] * UserList[k][Resource_Demand_loc ] * 15
        const[k][USERCOUNT + k] = 1
    
    return const

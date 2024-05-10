
def Default_UtilityFunction(USERCOUNT, user_list, OBJSIZE):
    const = [[0] * OBJSIZE for _ in range(USERCOUNT)]
    
    for k in range(USERCOUNT):
        const[k][k] = -user_list[k]  # x_k * r_k
        const[k][USERCOUNT + k] = 1  # U_k
    
    return const

import numpy as np

def reg2(USERCOUNT, OBJSIZE, user_list, R1, R2, U2, Hmax):
    const = np.zeros((USERCOUNT, OBJSIZE))
    for k in range(USERCOUNT):
        const[k, k] = -(user_list[k]*R1[k]*U2[k]) / (R2[k]-R1[k])  # xk * Rk
        const[k, USERCOUNT + k] = 1  # Uk
        const[k, 2*USERCOUNT + k] = Hmax  # Key1
        const[k, 3*USERCOUNT + k] = -Hmax  # Key2
    return const
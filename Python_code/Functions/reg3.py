import numpy as np

def reg3(USERCOUNT, OBJSIZE, user_list, U2, U3, R2, R3, Hmax):
    const = np.zeros((USERCOUNT, OBJSIZE))
    for k in range(USERCOUNT):
        const[k, k] = -user_list[k]*R2[k]*(U3[k]-U2[k]) / (R3[k]-R2[k])  # xk
        const[k, USERCOUNT + k] = 1  # Uk
        const[k, 2*USERCOUNT + k] = Hmax  # key1
        const[k, 3*USERCOUNT + k] = Hmax  # key2
        const[k, 4*USERCOUNT + k] = -Hmax  # key3
    return const
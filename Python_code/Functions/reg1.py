import numpy as np

def reg1(USERCOUNT, OBJSIZE, U1, Hmax):
    const = np.zeros((USERCOUNT, OBJSIZE))
    for k in range(USERCOUNT):
        const[k, USERCOUNT + k] = 1  # Uk
        const[k, 2*USERCOUNT + k] = U1[k] - Hmax  # key 1
    return const
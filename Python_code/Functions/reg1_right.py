import numpy as np

def reg1_right(USERCOUNT, U1):
    const = np.zeros(USERCOUNT)
    for k in range(USERCOUNT):
        const[k] = U1[k]
    return const
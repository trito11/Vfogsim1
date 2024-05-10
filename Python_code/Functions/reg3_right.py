import numpy as np

def reg3_right(USERCOUNT, U2, Hmax):
    const = np.zeros(USERCOUNT)
    for k in range(USERCOUNT):
        const[k] = U2[k] + 2 * Hmax
    return const
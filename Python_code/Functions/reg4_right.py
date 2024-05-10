import numpy as np

def reg4_right(USERCOUNT, Hmax):
    const = np.zeros(USERCOUNT)
    for k in range(USERCOUNT):
        const[k] = 2 * Hmax
    return const
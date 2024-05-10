import numpy as np

def reg2_right(USERCOUNT, Hmax):
    const = np.zeros(USERCOUNT)
    for k in range(USERCOUNT):
        const[k] = Hmax
    return const
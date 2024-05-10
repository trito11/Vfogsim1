import numpy as np
import numpy as np

def key_function(USERCOUNT, OBJSIZE, r, R1, R2, R3):
    key1_const = np.zeros((USERCOUNT, OBJSIZE))
    key2_const = np.zeros((USERCOUNT, OBJSIZE))
    key3_const = np.zeros((USERCOUNT, OBJSIZE))

    for k in range(USERCOUNT):
        key1_const[k, 2 * USERCOUNT + k] = 1
        
        # Xử lý phép chia cho 0
        if R1[k] != 0:
            key1_const[k, k] = -r[k] / R1[k]
        else:
            key1_const[k, k] = -100000

        key2_const[k, 3 * USERCOUNT + k] = 1
        # Xử lý phép chia cho 0
        if R2[k] != 0:
            key2_const[k, k] = -r[k] / R2[k]
        else:
            key2_const[k, k] = -100000

        key3_const[k, 4 * USERCOUNT + k] = 1
        # Xử lý phép chia cho 0
        if R3[k] != 0:
            key3_const[k, k] = -r[k] / R3[k]
        else:
            key3_const[k, k] = -100000

    key_const = np.vstack((key1_const, key2_const, key3_const))

    return key_const


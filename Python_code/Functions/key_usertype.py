import numpy as np

def key_usertype(USERCOUNT, R1, R2, R3):
    key1_const = np.zeros((USERCOUNT, 1))
    key2_const = np.zeros((USERCOUNT, 1))
    key3_const = np.zeros((USERCOUNT, 1))
    
    for k in range(USERCOUNT):
        key1_const[k, 0] = R1[k] / 100
        key2_const[k, 0] = R2[k] / 100
        key3_const[k, 0] = R3[k] / 100
    
    key_const = np.vstack((key1_const, key2_const, key3_const))
    
    return key_const

import numpy as np
from intlinprog_gurobi import intlinprog_gurobi
from  key_function import  key_function
from  Default_Bandwidth import Default_Bandwidth
from reg1 import reg1
from reg1_right import reg1_right
from reg1 import reg1
from reg1_right import reg1_right
from reg2 import reg2
from reg2_right import reg2_right
from reg3 import reg3
from reg3_right import reg3_right
from reg4 import reg4
from reg4_right import reg4_right
from memory_profiler import profile
import gc



def Default_Scheduler(user_list, upper_speed, R1, R2, R3, U1, U2, U3, window_size):
    USERCOUNT = user_list.shape[0]
    Hmax = 1000
    
    # Objective Function
    c = np.zeros(USERCOUNT * 2 + 3 * USERCOUNT)
    c[USERCOUNT:USERCOUNT * 2] = -1
    c[2 * USERCOUNT:] = -1
    
    OBJSIZE = c.shape[0]
    

    # Total assigned resources are limited with the available resources
    Tot_AssignedResources = np.concatenate((np.ones(USERCOUNT), np.zeros(OBJSIZE - USERCOUNT)))
    
    Val_AssignedResources = [1]

    # Upper limit per user
    Tot_AssignedBandwidth = Default_Bandwidth(20, OBJSIZE, USERCOUNT, user_list)
    
    Val_AssignedBandwidth = np.array(R3 )* 20
    
    
    # Sigmoid function design
    keys_constraint = key_function(USERCOUNT, OBJSIZE, user_list, R1, R2, R3)
    keys_lefthandside = np.zeros(USERCOUNT * 3)

    # Utility function
    reg1_const_left = reg1(USERCOUNT, OBJSIZE, U1, Hmax)
    reg1_const_right = reg1_right(USERCOUNT, U1)
    reg2_const_left = reg2(USERCOUNT, OBJSIZE, user_list, R1, R2, U2, Hmax)
    reg2_const_right = reg2_right(USERCOUNT, Hmax)
    reg3_const_left = reg3(USERCOUNT, OBJSIZE, user_list, U2, U3, R2, R3, Hmax)
    reg3_const_right = reg3_right(USERCOUNT, U2, Hmax)
    reg4_const_left = reg4(USERCOUNT, OBJSIZE, Hmax, U3)
    reg4_const_right = reg4_right(USERCOUNT, Hmax)
    
    # Optimizer
    
    B = np.vstack((
        Tot_AssignedResources,
        Tot_AssignedBandwidth,
        keys_constraint,
        reg1_const_left,
        reg2_const_left,
        reg3_const_left,
        reg4_const_left
    ))
    

    b =  np.hstack((Val_AssignedResources,
        Val_AssignedBandwidth,
        keys_lefthandside.ravel(),
        reg1_const_right.ravel(),
        reg2_const_right.ravel(),
        reg3_const_right.ravel(),
        reg4_const_right.ravel()
    ))
    
    lb = np.zeros(OBJSIZE)
    ub = np.ones(OBJSIZE) * 100
    ub[USERCOUNT * 2:] = 1
    intcon = list(range(USERCOUNT * 2, OBJSIZE))
    A=None
    a=None

    # Sử dụng hàm intlinprog_gurobi đã xây dựng ở trên
    xsol= intlinprog_gurobi(c, intcon, B, b, A, a, lb, ub)
    res = xsol[:USERCOUNT]
    return res

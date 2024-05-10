import numpy as np
import gurobipy as gp
from gurobipy import GRB
from memory_profiler import profile
import gc

# @profile
def intlinprog_gurobi(f, intcon, A, b, Aeq=None, beq=None, lb=None, ub=None):
    # Xác định số lượng biến
    if A is not None:
        n = A.shape[1]
    elif Aeq is not None:
        n = Aeq.shape[1]
    else:
        raise ValueError('No linear constraints specified')

    # Chuyển đổi ma trận ràng buộc sang định dạng tuplelist
    # Tạo mô hình Gurobi
    model = gp.Model()

    # Thiết lập hàm mục tiêu

    # Thiết lập loại biến
    vtype = ['C'] * n
    for i in intcon:
        vtype[i] = 'I'

    # Thêm biến vào mô hình
    x = model.addVars(n, vtype=vtype)
    model.setObjective(gp.quicksum(f[j] * x[j] for j in range(n)), GRB.MINIMIZE)
    
    # Thêm ràng buộc không bằng nhau
    if A is not None:
        for i in range(A.shape[0]):
            model.addConstr(gp.quicksum(A[i][j] * x[j] for j in range(n)) <= b[i])

    # Thêm ràng buộc bằng nhau
    if Aeq is not None:
        for i in range(Aeq.shape[0]):
            model.addConstr(gp.quicksum(Aeq[i][j] * x[j] for j in range(n)) == beq[i])

    # Thiết lập giới hạn dưới và giới hạn trên
    if lb is not None:
        model.setAttr('LB', x, lb)
    if ub is not None:
        model.setAttr('UB', x, ub)

    # Thiết lập tham số của trình giải
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 1800)

    # Tối ưu hóa mô hình
    model.optimize()
    if model.status == GRB.OPTIMAL:
        x_opt = [x[i].X for i in range(n)]
        
    else:
        print('No solution')

    model.dispose()
    del model
    gc.collect()
    # Trích xuất kết quả tối ưu hóa
    return x_opt

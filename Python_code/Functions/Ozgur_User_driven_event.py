import numpy as np

def Ozgur_User_driven_event(N_user, N_appType, T_totalSim, flag, Event, port):
    # Khởi tạo danh sách để lưu thông tin về sự kiện
   

    # Tạo ngẫu nhiên loại ứng dụng cho mỗi người dùng
    ui_taskType = np.random.randint(1, N_appType + 1, size=N_user)

    # Định nghĩa dữ liệu ứng dụng cho mỗi loại ứng dụng
    appData = np.zeros((3, 3))
    appData[0] = [2000, 250, 0.15]  # Light
    appData[1] = [2250, 2500, 0.20]  # Medium
    appData[2] = [2500, 10000, 0.25]  # Heavy
    N_cell = 1
    Epoch = 0.5
    Lambda = N_user / N_cell / Epoch
    t = -np.log(np.random.rand()) / Lambda
    n = -1
    # Tạo ngẫu nhiên một ma trận arc theo phân phối Poisson
    arc = np.random.poisson(N_user, size=N_user)
    arc = arc / np.max(arc)
    arc[arc >= Event * np.mean(arc)] = 1
    arc[arc < Event * np.mean(arc)] = 0

    ui = np.zeros(len(port))
    tt = np.zeros(len(port))
    ts = np.zeros(len(port))
    et = np.zeros(len(port))

    for k in range(len(port)):
        if port[k] > 0:
            n = n + 1
            if arc[n] > 0:
                ui[n] = k  # userId
                if flag == 0:
                    tt[n] = ui_taskType[int(ui[n])-1]  # single taskType
                else:
                    tt[n] = np.random.randint(1, N_appType + 1)  # multiple taskType
                ts[n] = appData[int(tt[n]-1)][0] + appData[int(tt[n]-1)][0] * (np.random.rand() * 0.2 - 0.1)  # taskSize
                et[n] = t
                t = t - np.log(np.random.rand()) / Lambda
    
    eventInfo = [et[:n+1],ui[:n+1],tt[:n+1],ts[:n+1]]
   

    return eventInfo

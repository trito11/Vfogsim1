import numpy as np
import pandas as pd
import sys
import os
import time
from pathlib import Path
link=Path(os.path.abspath(__file__))
link=link.parent.parent
link=os.path.join(link, "Functions")
sys.path.append(link)
print(link)
from preprocess import preprocess
from UserMatrix_Func_rl import UserMatrix_Func
from Ozgur_User_driven_event_rl import Ozgur_User_driven_event
from Scheduler import Scheduler
from Default_CellUpdate import Default_CellUpdate
 # assuming preprocess is a custom function or module
from config import *
from memory_profiler import profile
import torch
from numba import jit, cuda 
from metric import *
from geopy.distance import geodesic
from pyproj import Proj, transform




source = "100"  # Dataset
EventRate = 0.8  # Less is more! varies between 0.5 =>1.1
servicediff = 0
traffic = preprocess("Input.csv", source)
traffic = np.array(traffic)
traffic = np.delete(traffic, 0, 1)
traffic = np.delete(traffic, 0, 0)
# traffic[:, 1] += 1

bus = preprocess("bus.csv", source)
bus = np.array(bus)
bus = np.delete(bus, 0, 1)
bus = np.delete(bus, [2,3,4,7], 1)
# bus[:, :2] += 1
id_values = bus[:, 1]
bus_number=len(np.unique(id_values))
bus_dict=dict()
for id in range(bus_number):
    bus_dict[id]=bus[bus[:,1]==id]
NUM_VEHICLE=bus_number
# print(bus_dict.keys())

# Duration of simulation & users -- upperbounds
TimeFilter = 200
UserFilter = 500
Simulation_Duration = traffic[-1, 0]+1
USERCOUNT = max(traffic[:, 1])+1
CELLCOUNT = max(traffic[:, -1])
USERCOUNT = min(USERCOUNT, UserFilter)
Simulation_Duration = min(Simulation_Duration, TimeFilter)


# SIMULATION PARAMETERS

GAMMA = 0.000001
price_list = np.array([1, 2, 5, 10])  # Price of certain services
APP0_R = np.array([0.05, 0.07, 0.071])  # R1 R2 R3 %BG
APP1_R = np.array([0.01, 0.075, 0.4])  # R1 R2 R3 %M2M
APP2_R = np.array([0.1, 0.225, 0.55])  # R1 R2 R3 %Inelastic
APP3_R = np.array([0, 1.083, 20])  # R1 R2 R3 %Elastic
APP0_U = np.array([0, 1, 1])  # U1 U2 U3
APP1_U = np.array([-1, 0.7, 1])  # U1 U2 U3
APP2_U = np.array([-0.5, 0.7, 1])  # U1 U2 U3
APP3_U = np.array([0, 1, 1.8])  # U1 U2 U3
W = np.array([1, 1, 1, 1])  # Per App type

# AUXILIARY VARIABLES
SINR_loc = 0
SpectralResources_loc = 1
Demand_Time_loc = 2
Demand_Resource_loc = 3
Demand_ServiceType_loc = 4
Cell_ID_loc = 5
IsBlocked_loc = 6
Rem_Time_loc = 7
X_loc = 8
Y_loc = 9
Speed_loc = 10
s_in = 11
s_out = 12
Cell_Change_loc = 13
CELLCOUNT=int(CELLCOUNT)
USERCOUNT=int(USERCOUNT)
CellMatrix = np.zeros((CELLCOUNT, 3, Simulation_Duration),dtype=float)
B_pow = 100  # if it is 0 - no fog devices on buses

UserMatrix = UserMatrix_Func(USERCOUNT, Simulation_Duration, traffic, SINR_loc, Cell_ID_loc, X_loc, Y_loc, Speed_loc, 14)

ServiceRequirements = np.array([APP0_R, APP1_R, APP2_R, APP3_R])
ServiceUtilities = np.array([APP0_U, APP1_U, APP2_U, APP3_U])

# THE EVENT GENERATION ~ PER TTI
events = np.zeros((1, 4))
for n in range(Simulation_Duration):
    port = UserMatrix[:, Cell_ID_loc, n] > 0
    dummy = Ozgur_User_driven_event(np.sum(port, 0), 3, 1, 1, EventRate, UserMatrix[:, Cell_ID_loc, n] > 0)
    dummy = np.transpose(dummy)
    dummy = dummy[~np.any(dummy == 0, axis=1)]
    dummy[:, 0] = n
    events = np.vstack((events, dummy))
events = np.delete(events, 0, 0)
for i in range(events.shape[0]):
    time_slot = int(events[i, 0])
    user = int(events[i, 1])
    app_type = events[i, 2]
    demand = events[i, 3]
    UserMatrix[user, Demand_ServiceType_loc, time_slot] = app_type
    UserMatrix[user, Demand_Time_loc, time_slot] = np.random.randint(DEADLINE[0],DEADLINE[1])*100/100
    UserMatrix[user, Demand_Resource_loc, time_slot] = demand
    UserMatrix[user, s_in, time_slot]=np.random.randint(MIN_S_IN, MAX_S_IN)/1000
    UserMatrix[user, s_out, time_slot]=np.random.randint(MIN_S_OUT*10, MAX_S_OUT*10)/10000




import numpy as np
import pandas as pd
import gym
from gym import spaces
from gym.utils import seeding
import copy
import os
from metric import *
# from create_data import create_location_task_after
from MyGlobal import MyGlobals

class BusEnv(gym.Env):
    def __init__(self, env=None):
        self.env = env
        self.alpha=0.8
        # episode đầu
        self.index_of_episode = 0
        # Tổng phần thưởng
        self.sum_reward = 0
        # Số phần thưởng đã nhận
        
        # Không gian hành động
        self.n_actions=NUM_ACTION
        # Không gian state
        self.n_observations=NUM_STATE  

        try:
            os.makedirs(RESULT_DIR + MyGlobals.folder_name)
            print(MyGlobals.folder_name)
        except OSError as e:
            print(e)
        
        self.reward_files = open(
            RESULT_DIR + MyGlobals.folder_name + "reward.csv", "w")
        self.over_time_files = open(
            RESULT_DIR + MyGlobals.folder_name+ "over_time_task.csv", "w")
        self.delay_files = open(
            RESULT_DIR + MyGlobals.folder_name+ "delay.csv", "w")
        self.server_allocation = open(
            RESULT_DIR + MyGlobals.folder_name+ "server_allocation.csv", "w")
        self.delay_allocation = open(
            RESULT_DIR + MyGlobals.folder_name+ "delay_allocation.csv", "w")
        self.extra_allocation = open(
            RESULT_DIR + MyGlobals.folder_name+ "extra_allocation.csv", "w")
        self.tolerance_time_files = open(
            RESULT_DIR + MyGlobals.folder_name+ "sum_tolerance_time.csv", "w")
        
        tempstr = "local"
        for i in range(1, NUM_VEHICLE):
            tempstr += ",xe_" + str(i)
        
        self.server_allocation.write(tempstr + '\n')
        self.delay_allocation.write(tempstr + '\n')
        self.extra_allocation.write(tempstr + '\n')
        
        self.reward_files.write('reward,reward_accumulate\n')
        self.over_time_files.write('drop\n')
        self.delay_files.write('delay,delay_avg\n')
        self.tolerance_time_files.write('sum_tolerance_time\n')

        self.sum_reward = 0
        self.sum_reward_accumulate = 0
        self.sum_over_time = 0
        self.sum_delay = 0
        self.nreward = 0
        self.nstep = 0

    def readbus(self, number_bus, time):
        #đọc excel tính lat,lng của xe bus tại t=time
        data = self.data_bus[number_bus]

        after_time = data[data[:, 0] >= time]
        pre_time = data[data[:, 0] <= time]
        # if len(after_time) == 0:
        #     return 1.8,1.8
        las = after_time[0]
        first = pre_time[-1]
        diff1=las[0]-first[0]
        diff2=time-first[0]
        # weighted average of the distance
        if diff1==0:
            lat,lng=utm_to_latlon(first[2],first[3])
        else:
            lat,lng=calculate_intermediate_coordinate(utm_to_latlon(first[2],first[3]),utm_to_latlon(las[2],las[3]),diff2/diff1)
        return lat, lng


    def seed(self, seed=SEED):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def replay(self):
        # Khởi đâù của iter đặt lại episode về 0
        self.index_of_episode = 0
        self.data_bus=bus_dict


    def reset(self):
        # Khởi tạo state ban đầu 
        self.observation=np.zeros(NUM_STATE)
        self.sum_tolerance_time = 0
        self.sum_delay=0
        self.sum_over_time=0
        self.nstep=0
        # Đếm số hành động                                                                             
        self.n_tasks_in_node = [0] * (NUM_ACTION)

        self.n_tasks_delay_allocation = [0] * (NUM_ACTION)

        self.n_tasks_extra_allocation = [0] * (NUM_ACTION)
        
        #Đọc dữ liệu từ file task ứng với số rpisode
        self.data =  events
        #Tạo 1 queue lấy các task có cùng time với task ban đầu
        self.queue = copy.deepcopy(
                self.data[self.data[:, 0] == self.data[0][0]])
        # self.queue = self.queue[self.queue[:, 2].argsort()]
        self.data = self.data[self.data[:, 0] != self.data[0][0]]
        
        #Thời gian đầu episode
        self.time = self.queue[0][0]
        #Cập nhật khoảng cách đến từng xe 
        for i in range(NUM_VEHICLE):
            time=self.data_bus[i][:,0]
            min_time=time[0]
            max_time=time[-1]
            if self.time<min_time or self.time>max_time:
                self.observation[2 *
                                 i + 5]=10000
            else:
                
                lat,lng=self.readbus(i,self.time)
                print(int(self.time))
                task=UserMatrix[i,:,int(self.time)]
                # print(f"task:{task}")
                task_lat,task_lng=utm_to_latlon(task[X_loc],task[Y_loc])
                # print(task_lat,task_lng)
                self.observation[2 *
                                 i + 5] = distance_between_points(lat,lng,task_lat,task_lng)
    
        #Nếu ko phải episode đầu thì trừ hàng chờ đi độ lệch time giữa đít episode trc và đầu episode sau
        if self.index_of_episode!=0:
            for i in range(NUM_VEHICLE):
                self.observation[2 * i + 6] = max(
                    0, self.observation[2 * i + 6]-(self.time-self.time_last))
        #Thời gian cuối episode
        self.time_last = self.data[-1][0]
        self.task=UserMatrix[int(self.queue[0][1]),:,int(self.queue[0][0])]
                
        self.observation[0] = self.task[Demand_Resource_loc] #REQUIRED_GPU_FLOPS
        self.observation[1] = self.task[s_in] #s_in
        self.observation[2] = self.task[SINR_loc] #s_out
        self.observation[3] = self.task[Demand_Time_loc] #deadline
        self.observation[4] = self.queue[0][0]

        
        # Chỉ đến episode tiếp theo
        self.index_of_episode +=1
        #Trả về state đầu tiên
        return self.observation

    def step(self, action):
    #Action là số kiểu int ko phải [] hoặc tensor
        time_delay = 0
        drop_out=0

        #Nếu không drop
        if action > 0:
            #khoảng cách yêu cầu
            distance_req = self.observation[(action)*2+3]
            
            #hàng đợi cũ
            old_waiting_queue = self.observation[(action)*2+4]
            # print(old_waiting_queue)
            # print(f"hang cho:{old_waiting_queue}")
            # Rate_trans_req_data = 
            #thời gian truyền đi
            # print(self.observation[6:20:2])
            # print(self.observation)
            # print(f"Khoang cach:{distance_req}")
            # # print(f"Thoi gian truyen{self.observation[1]/(Rate_trans_req_data)}")
            # # print(f"Tốc độ xử lý :{self.observation[0] / PROCESSING_POWER }\n")
            Rate_trans_res_data=B_pow *np.log2(1 + self.task[SINR_loc] / GAMMA)
            # print(f"Thoi gian truyen{self.observation[1]/(Rate_trans_res_data)}")
            # print(f"Tốc độ xử lý :{self.observation[0] / PROCESSING_POWER }\n")
        

            # waiting queue                        # computation required / computation
            new_waiting_queue = self.observation[0] / PROCESSING_POWER       \
                + max(self.observation[1]/(Rate_trans_res_data), old_waiting_queue) #Độ lệch giữ hàng chờ và thời gian truyền đi
            
            #tọa độ xe sau khi xử lý task
            # after_lat_bus,after_lng_bus = self.readcsv(
            #     f'xe_{action}', new_waiting_queue+self.time)
        
            #Tọa độ người khi đó
            # task_lat,task_lng=create_location_task_after(self.queue[0][1],self.queue[0][2],new_waiting_queue)

            #Khoang cách lúc sau
            # distance_response=haversine(after_lat_bus,after_lng_bus,task_lat,task_lng)
            # Rate_trans_res_data = getRateTransData(channel_banwidth=CHANNEL_BANDWIDTH, pr=Pr, distance=distance_response,
            #                                        path_loss_exponent=PATH_LOSS_EXPONENT, sigmasquare=SIGMASquare)
            #Tính toán thời gian trễ
            # time_delay = new_waiting_queue + self.observation[2]/(Rate_trans_res_data)
            time_delay = new_waiting_queue 
            self.observation[(action)*2+4] = new_waiting_queue

        #nếu drop thì xử lý tại local
        else:
            drop_out=1 
            mobie_flops=random.randint(MOBIE_GPU_FLOPS_MIN,MOBIE_GPU_FLOPS_MAX)
            time_delay = self.observation[0]/mobie_flops


        self.n_tasks_in_node[action] = self.n_tasks_in_node[action]+1 #Hàm ghi lại các hành động
        self.n_tasks_delay_allocation[action] += time_delay #Hàm ghi lại tổng delay của mỗi xe
        self.sum_delay = self.sum_delay + time_delay #tổng delay


        extra_time = self.observation[3] - time_delay #thời gian thừa

        precent_time_finish=extra_time/self.observation[3] #Tỷ lệ thời gian thừa

        #tính toán phần thưởng
        reward_drop=-drop_out/EXPECTED_DROP

        if precent_time_finish>=0:
            reward_not_drop=0
        else:
            reward_not_drop=precent_time_finish

        reward = reward_not_drop

        # reward = self.alpha*reward_not_drop+ (1-self.alpha)* reward_drop

        # caculate tolerance time
        self.sum_tolerance_time += max(0, -extra_time)
        self.n_tasks_extra_allocation[action] += extra_time


        #Xóa task đã xử lý, cập nhật thêm các tác vụ mới xuấ hiện cùng time
        if len(self.queue) != 0:
            self.queue = np.delete(self.queue, (0), axis=0)

        #hàng chờ hết còn task ngoài lấy tiếp task đưa vào hàng chờ
        if len(self.queue) == 0 and len(self.data) != 0:
            self.queue = copy.deepcopy(
                self.data[self.data[:, 0] == self.data[0][0]])
            # self.queue = self.queue[self.queue[:, 2].argsort()]
            
            # position of cars
            
            #Độ lệch thời gian giữa 2 tác vụ
            time = self.data[0][0] - self.time
            #Cập nhật lại các hàng chờ
            for i in range(NUM_VEHICLE):
                self.observation[2 * i +
                                 6] = max(0, self.observation[2 * i + 6]-time)
            #Lấy thời gian hiện tại
            self.time = self.data[0][0]
            #Cập nhật các tác vụ còn lại
            self.data = self.data[self.data[:, 0] != self.data[0, 0]]
        #Còn task trong hàng chờ thì lấy dữ liệu tiếp do thời gian các task trong hàng đợi như nhau lên không cần cập nhật lại hàng chờ
        if len(self.queue) != 0:
            
            self.task=UserMatrix[int(self.queue[0][1]),:,int(self.queue[0][0])]
                
            self.observation[0] = self.task[Demand_Resource_loc] #REQUIRED_GPU_FLOPS
            self.observation[1] = self.task[s_in] #s_in
            self.observation[2] = self.task[SINR_loc] #s_out
            self.observation[3] = self.task[Demand_Time_loc] #deadline
            self.observation[4] = self.queue[0][0]
            
            for i in range(NUM_VEHICLE):
                time=self.data_bus[i][:,0]
                min_time=time[0]
                max_time=time[-1]
                if self.time<min_time or self.time>max_time:
                    self.observation[2 *
                                    i + 5]=1000000
                else:
                    
                    lat,lng=self.readbus(i,self.time)
                    task=UserMatrix[i,:,int(self.time)]
                    task_lat,task_lng=utm_to_latlon(task[X_loc],task[Y_loc])
                    self.observation[2 *
                                    i + 5] = distance_between_points(lat,lng,task_lat,task_lng)
                    


        # check end of episode?
        done = len(self.queue) == 0 and len(self.data) == 0
        self.sum_reward += reward
        if self.observation[3] < time_delay:
            self.sum_over_time += 1

        self.nreward += 1
        self.nstep += 1
        self.tolerance_time_files.write(str(self.sum_tolerance_time)+"\n")
        # check end of program? to close files
        avg_reward = self.sum_reward/self.nstep
        avg_reward_accumulate = self.sum_reward_accumulate/self.nreward
        self.reward_files.write(
            str(avg_reward)+','+str(avg_reward_accumulate)+"\n")
        self.over_time_files.write(str(self.sum_over_time/self.nstep)+"\n")
        self.delay_files.write(
            str(self.sum_delay)+','+str(self.sum_delay/self.nstep)+"\n")
        if done:
            print(self.n_tasks_in_node)

        #Ghi kết quả  ra các file

            
            tempstr = ','.join([str(elem) for elem in self.n_tasks_in_node])
            self.server_allocation.write(tempstr+"\n")
            tempstr = ','.join([str(elem/nb_step) if nb_step else '0' for elem, nb_step in zip(
                self.n_tasks_delay_allocation, self.n_tasks_in_node)])
            
            self.delay_allocation.write(tempstr+"\n")

            tempstr = ','.join([str(elem/nb_step) if nb_step else '0' for elem, nb_step in zip(
                self.n_tasks_extra_allocation, self.n_tasks_in_node)])
            self.extra_allocation.write(tempstr+"\n")
            
            
            # sum_tolerance time
            

            self.old_avg_reward = self.sum_reward/self.nstep
            self.sum_reward = 0
            self.nstep = 0
            self.sum_over_time = 0
            self.sum_delay = 0

        return self.observation, reward, done
if __name__ =="__main__":
    import matplotlib.pyplot as plt
    x=dict()
    for n in range(Simulation_Duration):
        time_slot = int(events[i, 0])
        if time_slot in x:
            x[time_slot]+=1
        else:
            x[time_slot]=1
    gia_tri = list(x.keys())
    so_lan = list(x.values())

    plt.bar(gia_tri, so_lan)
    plt.xlabel('Giá trị')
    plt.ylabel('Số lần xuất hiện')
    plt.title('Biểu đồ số lần xuất hiện của các giá trị')
    plt.show()


    

    

    


    




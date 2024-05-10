'''
File chạy thuật toán

Tạo class thuật toán

Tạo class memory: lưu trữ các state cũ  
    Có các hàm: 
        init: Tạo queue chứa
        push(state): Thêm state hiện tại vào hàng chờ
        sample(batch_size): lấy mẫu ngẫu nhiên n=batch_size mẫu từ hàng chờ để train

Tạo class Agent: đưa ra quyết định chọn hành động nào
    init(): Khai báo các trọng số sử dụng, batch_size, learning rate, khởi tạo các mạng, tạo hàm tối ưu, tạo hàng chờ
    select_action(state)=>đưa ra hành động: Chọn hành động
    optimal_model(các tham số để tối ưu): Tính toán loss, cập nhật lại các mạng để tối ưu hơn.
    train(số iter, số episode, số duration, tham số tối ưu,số episode):Huấn luyện thuật toán
    test(): 
    runAC():
'''
import sys
import os
from pathlib import Path
link=Path(os.path.abspath(__file__))
link=link.parent.parent
link=os.path.join(link, "system_model")
sys.path.append(link)
from itertools import count
import environment as env
from config import *
from MyGlobal import MyGlobals

class Agent_local:
    def __init__(self):
        self.env = env.BusEnv()

    def select_action(self):
        return 0 # 0 nghĩa là local
    
    def run(self, num_ep = NUM_EPISODE):
        self.env.replay()
        
        for ep in range(1):
            self.state = self.env.reset()

            done = False
            step = 0
            while (not done) and  (step := step + 1) :
                self.action = self.select_action()
                self.state, reward, done = self.env.step(self.action)

            print(f'Episode {ep}, avarage_reward: {self.env.old_avg_reward}\n')

if __name__ == '__main__':
    MyGlobals.folder_name="test/log_normal/local/"
    agent = Agent_local()
    agent.run(num_ep=90)
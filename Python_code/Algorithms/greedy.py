from itertools import count
import sys
import os
from pathlib import Path
link=Path(os.path.abspath(__file__))
link=link.parent.parent
link=os.path.join(link, "system_model")
sys.path.append(link)
import environment1 as env
from config import *
import torch
import random
from MyGlobal import MyGlobals

class DQNAgent:
    def __init__(self) :
        self.env=env.BusEnv()
        self.optimize=0
        self.env.seed(123)
        self.batch_size=2
        self.n_actions=NUM_ACTION
        self.n_observations=NUM_STATE

    def select_action(self,state,greedy):
        if greedy=='queue':
            a=True
            action=0
            for i in range(NUM_VEHICLE):
                
                if state[i*2+5]>0:
                    if a:
                        queue=state[i*2+6]
                        a=False
                        action=i+1
                    else:
                        if state[i*2+6]<=queue:
                            action=i+1
                            queue=state[i*2+6]
            
            return action
        if greedy=='best_sinr':
            a=True
            action=0
            for i in range(NUM_VEHICLE):
                if state[i*2+5]>0:
                    if a:
                        distance=state[i*2+5]
                        a=False
                        action=i+1
                    else:
                        if state[i*2+5]>=distance:
                            action=i+1
                            distance=state[i*2+5]
            return action
        if greedy=="Round_Robin":
            a=0
            while(True):
                i=random.randint(1,NUM_VEHICLE)
                if state[i*2+3]>0:
                    break
                else:
                    a+=1
                    if a>15:
                        return 0
            return i
 
    def run(self,greedy):
        
        self.env.replay()
        for episode in range(1):
            state = self.env.reset()
            done = False
            while not done:
                for i in count():
                    action = self.select_action(state,greedy)
                    next_state, reward, done= self.env.step(action)
                    if done:
                        next_state = None
                        print('Episode: {}, Score: {}'.format(
                            episode, self.env.old_avg_reward))
                        break
                    state = next_state
if __name__ == '__main__':
    greedy='Round_Robin'
    MyGlobals.folder_name=f"new_env/log_normal/{greedy}/"
    Agent=DQNAgent()
    Agent.run(greedy)
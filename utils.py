import os
import sys
import torch
import random
import numpy as np
from sumolib import checkBinary

def seed_everything(seed: int):
    """
    This function seeds every thing for reproducibility
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def sumo_configs(max_step, gui=False):
    """
    configuration of sumo environment
    """
    
    if 'SUMO_HOME' in os.environ:
        tools=os.path.join(os.environ['SUMO_HOME'],'tools')
        sys.path.append(tools)
    else: 
        sys.exit("please declare environment variable 'SUMO_HOME'")
   
    if gui:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')
    
    sumo_cmd = [sumoBinary, "-c", os.path.join("intersection", "env.sumocfg"),"--no-step-log", "True", "--waiting-time-memory", str(max_step) , "--log", "log.txt"]

    return sumo_cmd

def epsilon_schedule(step):
    progress = (np.clip(step / 540000, a_min=None, a_max=1)) # num of episodes * number of steps per episodes Ex: 100 * 5400, progress goes from 0 to 1
    epsilon = 1.0 - (1.0 - 0.01) * progress # 1 is the start value for epsilon and 0.01 is the end value for that
    return epsilon

if __name__ == "__main__":
    sumo_configs(100)
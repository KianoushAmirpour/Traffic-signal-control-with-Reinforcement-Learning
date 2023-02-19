import argparse
import time
import traci
import torch
import utils
from agent import Agent
from traffic_generator import RandomTraffic
from replay_buffer import ReplayBuffer
from utils import sumo_configs
from env import Environment
from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_cars_to_generate_per_episode', type=int, default=1000)
    parser.add_argument('--max_step_per_episode', type=int, default=5400)
    parser.add_argument('--gui', type=bool, default=False)
    parser.add_argument('--num_cells_for_edge', type=int, default=10)
    parser.add_argument('--device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--seed', type=int, default=14563)
    parser.add_argument('--input_channels', type=int, default=4, help="number of features to represents state")
    parser.add_argument('--num_actions', type=int, default=4)
    parser.add_argument('--buffer_max_size', type=int, default=1000000) 
    parser.add_argument('--buffer_min_size', type=int, default=1000) 
    parser.add_argument('--yellow_duration', type=int, default=4)
    parser.add_argument('--green_duration', type=int, default=10)
    parser.add_argument('--total_num_episode', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--update_target_step', type=int, default=1000)
    parser.add_argument('--num_epochs', type=int, default=1000) 
    parser.add_argument('--gamma', type=float, default=0.99)     
    parser.add_argument('--TAU', type=float, default=0.001)
    
    args = parser.parse_args()
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)
    
    return training_config

def train_dqn(configs):
    
    sumo_cmd = utils.sumo_configs(configs["max_step_per_episode"], configs["gui"])
    
    traffic_generator = RandomTraffic(configs["num_cars_to_generate_per_episode"], configs["max_step_per_episode"])           
    
    agent = Agent(configs["input_channels"], configs["num_actions"], configs["learning_rate"], configs["gamma"], 
                  configs["device"], configs["buffer_max_size"], configs["buffer_min_size"], configs["batch_size"])
    
    env = Environment(traffic_generator, sumo_cmd, configs["max_step_per_episode"], configs["num_cells_for_edge"],
                      configs["num_actions"], configs["yellow_duration"], configs["green_duration"])
    
    for episode in range(1, configs["total_num_episode"] + 1):
        
        print(f'\n{"-"*150} \n')
        
        start_time = time.time()
        
        epsilon = 1.0 - (episode / configs["total_num_episode"])
        
        # start of the simulation ---> generate traffic and traci.start
        env.start(episode)
        
        # variables to store the results of each episodes
        old_total_wait = 0
        old_state = -1
        old_action = -1
        
        waiting_times = {}
        sum_waiting_time = 0
        sum_queue_length = 0
        sum_neg_reward= 0
        cumulative_waiting_times = 0
        
        for step in range(1, configs["max_step_per_episode"] + 1):
            # getting the state at each step
            current_state = env.get_state_observation()
             
             # action based on the current state
            action = agent.act(current_state, epsilon)
             
            current_total_waiting_time = env.get_waiting_time(waiting_times) + env.get_queue_length()
            cumulative_waiting_times += current_total_waiting_time
            reward = old_total_wait - current_total_waiting_time
             
            if step != 1:
                agent.step(old_state, old_action, reward, current_state)
                
            if step != 1 and old_action != action:
                env.set_yellow_phase(old_action)
                env.simulate(step, configs["max_step_per_episode"], configs["yellow_duration"], sum_waiting_time, sum_queue_length)
            
            env.set_green_phase(action)
            env.simulate(step, configs["max_step_per_episode"], configs["green_duration"], sum_waiting_time, sum_queue_length)
            
            old_state = current_state
            
            old_action = action
            
            old_total_wait = current_total_waiting_time
            
            if reward < 0:
                sum_neg_reward += reward
        
        end_time = time.time()       
        print(f"Episode: {episode}/{configs['total_num_episode']} | Negative_reward : {sum_neg_reward} | cumulative_waiting_times (sec): {cumulative_waiting_times} | simulation_time (min) : {round((end_time - start_time)/60, 3)}")
        # saving results
        traci.close()
        
        # start of training
        # agent.learn(experiences, Gamma)
        
        
            
            
             
             
             
             
            
            
            
            
            
        
        
        
    
    
    
    
    
    
    # for episode in range(1, n_episodes):
    #     state = env.reset()
    #     score = 0
    #     for i in range(max_t):
    #         action = agent.act(state, eps_start) # FEEDFORWARD THE STATE AND GET THE Q VALUES , do epsilon greedy and get the action
    #         next_state , reward, done, _ = env.step(action)
    #         agent.step(state, action, reward, next_state, done)
    #         state = next_state
    #         score += reward
    #         if done:
    #             break
            
            
            
            
if __name__ == "__main__":
    
    configs = get_args()
    # writer = SummaryWriter(f"logs/{experiment_name}")
    train_dqn(configs)
    
    
    
    
    
        
     
    
    
    
    
    
    
    
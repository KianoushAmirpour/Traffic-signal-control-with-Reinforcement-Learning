import argparse
import time
import traci
import torch
import utils
from agent import Agent
from traffic_generator import RandomTraffic
from replay_buffer import ReplayBuffer
from sumo_env import Environment
from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gui', type=bool, default=False, help="whether to open the sumo-gui or not")
    
    parser.add_argument('--num_cars_to_generate_per_episode', type=int, default=1000)
    parser.add_argument('--max_step_per_episode', type=int, default=5400)
    parser.add_argument('--total_num_episode', type=int, default=50)
    
    parser.add_argument('--num_cells_for_edge', type=int, default=10, help="discretizing the environment space, considering 10 cells for each incomming lanes")
    parser.add_argument('--input_channels', type=int, default=4, help="number of features to represents the state: num of cars, average speed, waiting times, number of queued cars")
    parser.add_argument('--num_actions', type=int, default=4, help="NS-green, NSL-green, EW-green, EWL-green")
    
    parser.add_argument('--buffer_max_size', type=int, default=1000000) 
    parser.add_argument('--buffer_min_size', type=int, default=1000) 
    
    parser.add_argument('--yellow_duration', type=int, default=4)
    parser.add_argument('--green_duration', type=int, default=10)
    
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--update_model_weight', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--update_target_step', type=int, default=10000)
    parser.add_argument('--gamma', type=float, default=0.99)     
    
    parser.add_argument('--device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--seed', type=int, default=1111 ,help="for reproducibility")
    
    parser.add_argument('--save_model_freq', type=int, default=10000 ,help="saving the model after this steps")
    
    args = parser.parse_args()
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)
    return training_config

def train_dqn(configs):
    
    # utils.seed_everything(configs["seed"])
    
    # initializing the tensorboard for loging the stats
    writer = SummaryWriter("logs")
    
    # getting the configuration for sumo to be run(whether to open the interface and the path to env.sumocfg file)
    sumo_cmd = utils.sumo_configs(configs["max_step_per_episode"], configs["gui"])
    
    traffic_generator = RandomTraffic(configs["num_cars_to_generate_per_episode"], configs["max_step_per_episode"])           
    
    agent = Agent(configs["input_channels"], configs["num_actions"], configs["device"], configs["learning_rate"], configs["gamma"], 
                  configs["batch_size"], configs["update_model_weight"], configs["update_target_step"], writer,
                  configs["buffer_max_size"], configs["buffer_min_size"], configs["save_model_freq"]
                  )
    
    env = Environment(traffic_generator, sumo_cmd, configs["max_step_per_episode"], configs["num_cells_for_edge"],
                      configs["num_actions"], configs["yellow_duration"], configs["green_duration"]
                      )
    
    count_steps_for_epsilon = 0 # this will help to decay the epsilon
    
    for episode in range(1, configs["total_num_episode"] + 1):
        
        print(f'Episode: {episode}/{configs["total_num_episode"]}{"-"*60}')
        
        start_time = time.time()
        
        # start of the simulation ---> generate traffic and traci.start(sumo_cmd)
        env.start(episode)
        
        # variables to store the stats of each episodes
        old_total_wait = 0
        old_state = -1
        old_action = -1
        
        waiting_times = {}
        sum_neg_reward= 0
        cumulative_waiting_times = 0
        
        for step in range(1, configs["max_step_per_episode"] + 1):
            # getting the state representation at each step
            current_state = env.get_state_observation() # shape 4, 24, 24
            
            epsilon = utils.epsilon_schedule(count_steps_for_epsilon)
            # action based on the current state
            action = agent.act(current_state, epsilon) # output is an intiger indicating the action that has been chosen, ex: 0, 1, 2, 3
             
            current_total_waiting_time = env.get_waiting_time(waiting_times) + env.get_queue_length()
            cumulative_waiting_times += current_total_waiting_time
            reward = old_total_wait - current_total_waiting_time
             
            if step != 1:
                agent.step(old_state, old_action, reward, current_state)
                
            if step != 1 and old_action != action:
                env.set_yellow_phase(old_action)
                env.simulate(step, configs["max_step_per_episode"], configs["yellow_duration"])
            
            env.set_green_phase(action)
            env.simulate(step, configs["max_step_per_episode"], configs["green_duration"])
            
            old_state = current_state
            
            old_action = action
            
            old_total_wait = current_total_waiting_time
            
            if reward < 0:
                sum_neg_reward += reward
            
            count_steps_for_epsilon += 1
        
        writer.add_scalar("Negative Reward Per Episode/episodes", sum_neg_reward, episode)
        writer.add_scalar("Cumulative Waiting Time/episodes", cumulative_waiting_times, episode)
        writer.add_scalar("epsilon/episodes", epsilon, episode)
        
        end_time = time.time()   
        print(f"simulation time : {round((end_time - start_time)/60, 3)}\n")    
        traci.close()
    
    # save the last model
    torch.save({'model_state_dict': agent.dqn.state_dict(), 'optimizer_state_dict': agent.optimizer.state_dict()},
                f"checkpoints/last_dqn.pth")  
       
if __name__ == "__main__":
    
    configs = get_args()
    
    train_dqn(configs)
    
    
    
    
    
        
     
    
    
    
    
    
    
    
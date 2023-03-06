import argparse
import time
import traci
import torch
import utils
from model import DQN
from traffic_generator import RandomTraffic
from sumo_env import Environment


def get_args():
    parser = argparse.ArgumentParser()
    # traffic generation params
    parser.add_argument('--num_cars_to_generate_per_episode', type=int, default=2000)
    parser.add_argument('--max_step_per_episode', type=int, default=5400)
    # state representation params
    parser.add_argument('--num_cells_for_edge', type=int, default=10, help="discretizing the environment space, considering 10 cells for each incomming lanes")
    parser.add_argument('--input_channels', type=int, default=4, help="number of features to represents the state: num of cars, average speed, waiting times, number of queued cars")
    parser.add_argument('--num_actions', type=int, default=4, help="NS-green, NSL-green, EW-green, EWL-green")
    # simulation params
    parser.add_argument('--total_num_episode', type=int, default=5)
    parser.add_argument('--yellow_duration', type=int, default=4)
    parser.add_argument('--green_duration', type=int, default=10)
    parser.add_argument('--gui', type=bool, default=False, help="whether to open the sumo-gui or not")
    # general 
    parser.add_argument('--device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--seed', type=int, default=321654987 ,help="for reproducibility")
    
    args = parser.parse_args()
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)
    return training_config

def evaluate_dqn(configs):
    
    path_to_model = "./checkpoints/2000_5400_10_4_4_1000000_1000_100_4_10_False_32_500_20_0.0002_125_0.95_5_Huber_ADAM_10000_0.001_True_1111_DDQN/last_dqn.pth"
    model = DQN(configs["input_channels"], configs["num_actions"]).to(configs["device"])
    checkpoint = torch.load(path_to_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda().eval()
    
    # get the path for saving the logs and checkpoints based on the name of experiences
    plots_path = "./plots/2000_5400_10_4_4_1000000_1000_100_4_10_False_32_500_20_0.0002_125_0.95_5_Huber_ADAM_10000_0.001_True_1111_DDQN"
    
    # seed everything
    utils.seed_everything(configs["seed"])

    # getting the configuration for sumo to be run(whether to open the interface and the path to env.sumocfg file)
    sumo_cmd = utils.sumo_configs(configs["max_step_per_episode"], configs["gui"])
    
    # will be passed to the Environment class to generate random traffic for each episode
    traffic_generator = RandomTraffic(configs["num_cars_to_generate_per_episode"], configs["max_step_per_episode"])           
    
    # will simulate the whole process in sumo 
    env = Environment(
                    traffic_generator, sumo_cmd, configs["max_step_per_episode"], configs["num_cells_for_edge"],
                    configs["num_actions"], configs["yellow_duration"], configs["green_duration"]
                     )
    
    # count_steps_for_epsilon = 0 # this will help decay the epsilon
    negative_rewards_list = []
    cumulative_waiting_time_list = []
    
    # start of the simulation for each episode ---> generate random traffic and traci.start(sumo_cmd)
    
    for episode in range(1, configs["total_num_episode"] + 1):
        
        print(f'Episode: {episode}/{configs["total_num_episode"]} {"-"*60}')
        
        env.start(configs["seed"] + episode)
    
        # variables to store the statistics of each episodes
        old_total_wait = 0
        old_state = -1 # will store the state  ----> (state, action, reward, new state)
        old_action = -1 # will store the action  ----> (state, action, reward, new state)
        
        sum_neg_reward = 0 # cumulative negative reward for each episode
        cumulative_waiting_times = 0 # cumulative waiting time by all vehicles in each episode
        step = 0
        
        while step < configs["max_step_per_episode"]:
        # for step in range(1, configs["max_step_per_episode"] + 1):
            
            # getting the state representation at each step (a matrix consist of 4 dimension for num of cars, average speed, waiting times, number of queued cars)
            current_state = env.get_state_observation() #  output'shape : (4, 24, 24)
            
            #action from model or agent !!!!!!!!!!!!!!!
            state = torch.from_numpy(current_state).float().unsqueeze(0).to(configs["device"])
            action_values = model(state)
            action = action_values.argmax(dim=1)[0].cpu().numpy()
                
            # calculating the reward as a sum of waiting times (for every car) and lenght of the queues in the incoming roads
            current_total_waiting_time = env.get_waiting_time() + env.get_queue_length()
            cumulative_waiting_times += current_total_waiting_time
            reward = old_total_wait - current_total_waiting_time
                    
            # activating the yellow phase if it is needed    
            if step != 0 and old_action != action:
                env.set_yellow_phase(old_action)
                step = env.simulate(step, configs["max_step_per_episode"], configs["yellow_duration"])
            
            # activating the phase that was selected
            env.set_green_phase(action)
            step = env.simulate(step, configs["max_step_per_episode"], configs["green_duration"])

            old_state = current_state
            
            old_action = action
            
            old_total_wait = current_total_waiting_time
            
            # saving the meaningful rewards to see if agent is learning or not
            if reward < 0:
                sum_neg_reward += reward
            
        # logging the statistics of the episode to tensorboard
        negative_rewards_list.append(sum_neg_reward)
        cumulative_waiting_time_list.append(cumulative_waiting_times)
        
        traci.close()
    
    print(f"Test is done")  
        
    utils.plot(range(configs["total_num_episode"]), negative_rewards_list, "episodes", "Negative_rewards", "Negative rewards for all vehicles",
               "blue", f"{plots_path}/test_Negative_rewards.png")
    
    utils.plot(range(configs["total_num_episode"]), cumulative_waiting_time_list, "episodes", "cumulative_waiting_times", "cumulative waiting times for all vehicles",
               "blue", f"{plots_path}/test_cumulative_waiting_times.png")
    
    
if __name__ == "__main__":
    
    configs = get_args()
    
    evaluate_dqn(configs)
    
    
    
    
    
        
     
    
    
    
    
    
    
    
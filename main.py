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
    # traffic generation params
    parser.add_argument('--num_cars_to_generate_per_episode', type=int, default=2000)
    parser.add_argument('--max_step_per_episode', type=int, default=5400)
    # state representation params
    parser.add_argument('--num_cells_for_edge', type=int, default=10, help="discretizing the environment space, considering 10 cells for each incomming lanes")
    parser.add_argument('--input_channels', type=int, default=4, help="number of features to represents the state: num of cars, average speed, waiting times, number of queued cars")
    parser.add_argument('--num_actions', type=int, default=4, help="NS-green, NSL-green, EW-green, EWL-green")
    # memory params
    parser.add_argument('--buffer_max_size', type=int, default=1000000) 
    parser.add_argument('--buffer_min_size', type=int, default=1000) 
    # simulation params
    parser.add_argument('--total_num_episode', type=int, default=100)
    parser.add_argument('--yellow_duration', type=int, default=4)
    parser.add_argument('--green_duration', type=int, default=10)
    parser.add_argument('--gui', type=bool, default=False, help="whether to open the sumo-gui or not")
    # training params
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--update_model_weight', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--update_target_step', type=int, default=125)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--grad_clip', type=float, default=5)
    parser.add_argument('--loss_function', type=str, choices=["Huber", "MSE"], default="Huber")
    parser.add_argument('--optimizer', type=str, choices=["ADAM", "RMSprop"], default="ADAM")
    parser.add_argument('--save_model_freq', type=int, default=10000 ,help="saving the model after this steps")
    parser.add_argument('--TAU', type=float, default=0.001 ,help="how to update target network")
    parser.add_argument('--DDQN', type=bool, default=True)
    # general 
    parser.add_argument('--device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--seed', type=int, default=1111 ,help="for reproducibility")
    parser.add_argument('--text', type=str, default="diff_init")
    args = parser.parse_args()
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)
    return training_config

def train_dqn(configs):
    
    # check if directories for saving the logs and checkpoints exist, if not make them.
    utils.make_dir()
    
    # get the path for saving the logs and checkpoints based on the name of experiences
    checkpoints_path = utils.get_dir_name(configs, "checkpoints") 
    logs_path = utils.get_dir_name(configs, "logs")
    plots_path = utils.get_dir_name(configs, "plots")
    
    # seed everything
    # utils.seed_everything(configs["seed"])

    # initializing the tensorboard for loging the statistics
    writer = SummaryWriter(logs_path)
    
    # getting the configuration for sumo to be run(whether to open the interface and the path to env.sumocfg file)
    sumo_cmd = utils.sumo_configs(configs["max_step_per_episode"], configs["gui"])
    
    # will be passed to the Environment class to generate random traffic for each episode
    traffic_generator = RandomTraffic(configs["num_cars_to_generate_per_episode"], configs["max_step_per_episode"])           
    
    # will handle action selection, storing the results in memory and learning process of the agent
    agent = Agent(
                configs["input_channels"], configs["num_actions"], configs["device"], configs["learning_rate"],
                configs["gamma"], configs["batch_size"], configs["update_model_weight"], configs["update_target_step"],
                writer, configs["buffer_max_size"], configs["buffer_min_size"], configs["save_model_freq"],
                configs["grad_clip"], checkpoints_path, configs["TAU"], configs["DDQN"], optimizer=configs["optimizer"], loss_function=configs["loss_function"], 
                 )
    
    # will simulate the whole process in sumo 
    env = Environment(
                    traffic_generator, sumo_cmd, configs["max_step_per_episode"], configs["num_cells_for_edge"],
                    configs["num_actions"], configs["yellow_duration"], configs["green_duration"]
                     )
    
    # count_steps_for_epsilon = 0 # this will help decay the epsilon
    negative_rewards_list = []
    cumulative_waiting_time_list = []
    
    start_time_whole_process = time.time()
    for episode in range(1, configs["total_num_episode"] + 1):
        
        print(f'Episode: {episode}/{configs["total_num_episode"]} {"-"*60}')
        
        start_time = time.time()
        
        # start of the simulation for each episode ---> generate random traffic and traci.start(sumo_cmd)
        env.start(episode)
        
        # variables to store the statistics of each episodes
        old_total_wait = 0
        old_state = -1 # will store the state  ----> (state, action, reward, new state)
        old_action = -1 # will store the action  ----> (state, action, reward, new state)
        
        sum_neg_reward = 0 # cumulative negative reward for each episode
        cumulative_waiting_times = 0 # cumulative waiting time by all vehicles in each episode
        step = 0
        epsilon = 1.0 - (episode / configs['total_num_episode'])
        
        while step < configs["max_step_per_episode"]:
        # for step in range(1, configs["max_step_per_episode"] + 1):
            
            # getting the state representation at each step (a matrix consist of 4 dimension for num of cars, average speed, waiting times, number of queued cars)
            current_state = env.get_state_observation() #  output'shape : (4, 24, 24)
            
            # set the epsilon to be used in epsilon-greedy
            # epsilon = utils.epsilon_schedule(count_steps_for_epsilon)
            
            # select the action based on the current state and epsilon, output is an intiger indicating the action that has been chosen, ex: 0, 1, 2, 3
            action = agent.act(current_state, epsilon)
             
            # calculating the reward as a sum of waiting times (for every car) and lenght of the queues in the incoming roads
            current_total_waiting_time = env.get_waiting_time() + env.get_queue_length()
            cumulative_waiting_times += current_total_waiting_time
            reward = old_total_wait - current_total_waiting_time
            
            # storing the experiences in the memory and if there are enough samples, will update the weights of the model
            if step != 0:
                # agent.step(old_state, old_action, reward, current_state)
                agent.store_experience(old_state, old_action, reward, current_state, step)
            
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
            
            # count_steps_for_epsilon += 1
        
        # logging the statistics of the episode to tensorboard
        negative_rewards_list.append(sum_neg_reward)
        cumulative_waiting_time_list.append(cumulative_waiting_times)
        
        writer.add_scalar("Negative Reward Per Episode/episodes", sum_neg_reward, episode)
        writer.add_scalar("Cumulative Waiting Time/episodes", cumulative_waiting_times, episode)
        writer.add_scalar("epsilon/episodes", epsilon, episode)
        
        end_time = time.time()   
        print(f"simulation time for this episode: {round((end_time - start_time)/60, 3)}\n")  
        
        # end of the episode  
        traci.close()

        # training
        start_time = time.time()
        for epoch in range(configs["epochs"]):
            agent.step()
        end_time = time.time() 
        print(f"Training time for this episode: {round((end_time - start_time)/60, 3)}\n")
    
    # save the last model
    torch.save({'model_state_dict': agent.dqn.state_dict(), 'optimizer_state_dict': agent.optimizer.state_dict()},
                f"{checkpoints_path}/last_dqn.pth") 
     
    end_time_whole_process = time.time() 
    print(f"simulation time for all episodes: {round((end_time_whole_process - start_time_whole_process)/60, 3)}\n")
    
    utils.plot(range(configs["total_num_episode"]), negative_rewards_list, "Episodes", "Negative_rewards", "Negative rewards for all vehicles",
               "blue", f"{plots_path}/Negative_rewards.png")
    
    utils.plot(range(configs["total_num_episode"]), cumulative_waiting_time_list, "Episodes", "cumulative_waiting_times", "cumulative waiting times for all vehicles",
               "blue", f"{plots_path}/cumulative_waiting_times.png")
    
    
    
if __name__ == "__main__":
    
    configs = get_args()
    
    train_dqn(configs)
    
    
    
    
    
        
     
    
    
    
    
    
    
    
# Traffic signal control with Reinforcement Learning
This repository showcases my attempts to implement a Deep Q-Network (DQN) to enhance the efficiency and performance of a single traffic signal.

The image below illustrates the characteristics of an intersection in this project: a four-way intersection with three lanes per leg and protected left turns

![intersection](https://github.com/KianoushAmirpour/Traffic-signal-control-with-Reinforcement-Learning/assets/112323618/b049a3f9-5dc3-44f9-8324-88d7670ff5d1)

## Project structure
- model.py: The DQN class provides a Convolutional Neural Network model used in this project.
- agent.py: The Agent class manages the process of adding experiences to memory, updating model weights and target networks, and handling exploration-exploitation strategies.
- replay_buffer.py: The ReplayBuffer class is responsible for retrieving samples from memory and adding new samples to memory.
- sumo_env.py : This file interfaces with the `SUMO` traffic simulation tool, collecting state observations from the simulation. The Environment class initiates the simulation, extracting metrics such as the    number of cars, average speeds, queues, and waiting times, and presenting them in a matrix format
- evaluate.py : The evaluate script assesses the performance of the trained model on a test dataset.
- traffic_generator.py : The RandomTraffic class generates randomized traffic patterns using a Weibull distribution, allocating 70 percent of traffic for straight movements and 30 percent for turns.
- main.py :  The main script brings all components together for model training and preservation. It calculates rewards, advances simulation steps, and manages model saving.
- utils.py : The utils module offers a collection of useful helper functions to support various aspects of the project.
## Training Configuration
`--num_cars_to_generate_per_episode` (default: 2000): Number of cars generated per episode.  
`--max_step_per_episode` (default: 5400): Maximum number of steps per episode.  
## State Representation Parameters
`--num_cells_for_edge` (default: 10): Discretization of environment space, considering 10 cells for each incoming lane.  
`--input_channels` (default: 4): Number of features used to represent the state (num of cars, average speed, waiting times, queued cars).  
`--num_actions` (default: 4): Number of actions (NS-green, NSL-green, EW-green, EWL-green).  
## Memory Parameters
`--buffer_max_size` (default: 1000000): Maximum size of the replay buffer.  
`--buffer_min_size` (default: 1000): Minimum size required before starting training.  
## Simulation Parameters
`--total_num_episode` (default: 100): Total number of training episodes.  
`--yellow_duration` (default: 4): Duration of the yellow phase.  
`--green_duration` (default: 10): Duration of the green phase.  
`--gui` (default: False): Whether to open the SUMO GUI.  
## Training Parameters
`--batch_size` (default: 32): Batch size for training.  
`--epochs` (default: 500): Number of training epochs.  
`--update_model_weight` (default: 20): Frequency of model weight updates.  
`--learning_rate` (default: 0.0002): Learning rate for the optimizer.  
`--update_target_step` (default: 125): Frequency of target network updates.  
`--gamma` (default: 0.95): Discount factor for future rewards.  
`--grad_clip` (default: 5): Gradient clipping threshold.  
`--loss_function` (default: Huber): Choice of loss function (Huber/MSE).  
`--optimizer` (default: ADAM): Choice of optimizer (ADAM/RMSprop).  
`--save_model_freq` (default: 10000): Model saving frequency in steps.  
`--TAU` (default: 0.001): Rate for updating target network.  
`--DDQN` (default: False): Enable Double DQN if True.  
## General
`--device` (default: cuda:0 if available, else cpu): Device for computation.  
`--seed` (default: 1111): Seed for reproducibility.  
`--text` (default: diff_init): Additional text identifier.  
## Installation
To get started with this project, ensure you have the following dependencies installed:
 - PyTorch
 - NumPy
 - TraCI
 - Sumolib
## Acknowledgements
- [pytorch-learn-reinforcement-learning](https://github.com/gordicaleksa/pytorch-learn-reinforcement-learning)
- [Deep-QLearning-Multi-Agent-Perspective-for-Traffic-Signal-Control](https://github.com/GameDisplayer/Deep-QLearning-Multi-Agent-Perspective-for-Traffic-Signal-Control)


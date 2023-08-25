# Traffic-signal-control-with-Reinforcement-Learning
This repository showcases my exploration into leveraging Deep Q-Network (DQN) for optimizing a single traffic signal. The project employs reinforcement learning techniques to enhance traffic signal performance and efficiency.

## Project structure
   - model.py : The DQN class provides a CNN model as a Neural Network
   - agent.py : the Agent class hanles the learning process
   - replay_buffer.py : the ReplayBuffer class
   - sumo_env.py : used sumo . The Environment class starts the simulation and gets the number of cars, average speeds, queues and waiting times and returns them in the form of a matrix. it also handles to create a new step in simulation process.
   - evaluate.py : this file evaluates the trained model on a test set.
   - traffic_generator.py : the RandomTraffic class generates random traffic based on wiebull distribution. 70 percent for striagn and 30 for turns.
   - main.py : brings every thing together to train the model and save them. rewards????
   - utils.py : it has some usfu helper functions.
## Workflow
  folders creation, defining intersections (four armed legs. image), the process.
## args just run main.py?????



## Installation
To get started with this project, ensure you have the following dependencies installed:
 - PyTorch
 - NumPy
 - TraCI
 - Sumolib
## Acknowledgements
- [pytorch-learn-reinforcement-learning](https://github.com/gordicaleksa/pytorch-learn-reinforcement-learning)
- [Deep-QLearning-Multi-Agent-Perspective-for-Traffic-Signal-Control](https://github.com/GameDisplayer/Deep-QLearning-Multi-Agent-Perspective-for-Traffic-Signal-Control)


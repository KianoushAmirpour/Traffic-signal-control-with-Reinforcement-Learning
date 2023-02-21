import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import DQN
from replay_buffer import ReplayBuffer

class Agent:
    def __init__(self, input_channels, num_actions, device, learning_rate, gamma, batch_size,
                 update_model_weight, target_update_step, writer, buffer_max_size, buffer_min_size, save_model_freq):

        self.input_channels = input_channels
        self.num_actions = num_actions
        self.device = device
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_model_weight = update_model_weight
        self.target_update_step = target_update_step
        self.writer = writer
        self.buffer_max_size = buffer_max_size
        self.buffer_min_size = buffer_min_size
        
        self.memory = ReplayBuffer(self.buffer_max_size, self.buffer_min_size, self.batch_size, self.device)
        
        self.dqn = DQN(self.input_channels, self.num_actions).to(self.device)
        self.target_dqn = DQN(self.input_channels, self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.learning_rate)
        self.loss = nn.HuberLoss() 
        self.checkpoint_freq_save = save_model_freq
        
        self.losses = [] # to store the values of loss function
        self._update_model_weight_step = 0 # to count the steps that loss functions is being updated
        self._update_target_network_step = 0 # to count the steps that is needed to update the target network
        self.best_reward = -np.inf 
        
    def step(self, state, action, reward, next_state):
        self.memory.add_experience(state, action, reward, next_state)
        
        if self._update_model_weight_step %  self.update_model_weight == 0: # update model's weights every self.update_model_weight step(Ex: every 12 steps)
            if len(self.memory)> self.batch_size: # we update the model's weights when we have enough samples
                experiences = self.memory.get_sample_from_memory()
                self.learn(experiences)
        
        self._update_model_weight_step += 1
        # preodically save the model
        if self._update_model_weight_step % self.checkpoint_freq_save == 0:
            torch.save({'model_state_dict': self.dqn.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()},
                        f"checkpoints/dqn_update_weight_step_{self._update_model_weight_step}.pth")
            
    def act(self, state, epsilon):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device) # input's shape (4, 24, 24) ----> output's shape (1, 4, 24, 24)
        self.dqn.eval() # bc it is just a feedforward without a need to backpropagation
        with torch.no_grad():
            action_values = self.dqn(state) # output's shape : (1, 4)
        self.dqn.train()
        
        if random.random() < epsilon:
            return random.randint(0, self.num_actions -1) # an intiger indicating the action that has been chosen (Explore)
        else:
            return action_values.argmax(dim=1)[0].cpu().numpy() # an intiger indicating the action that has been chosen (Exploit)
        
    def learn(self, experiences):
        states, actions, rewards, next_states = experiences
        
        self.dqn.train()
        self.target_dqn.eval()

        current_state_q_values = self.dqn(states).gather(dim=1, index=actions) # output's shape = (bs, 1) , choose Q values that correspond to the actions were made in those states

        with torch.no_grad():
            next_state_max_q_values = self.target_dqn(next_states).max(dim=1, keepdim=True)[0] # output's shape : (bs , 1)
 
            target_q_values = rewards + (self.gamma *  next_state_max_q_values) # calculating the target values
        
        loss = self.loss(current_state_q_values, target_q_values).to(self.device)
        
        # logging the loss values to tensorboard
        self.writer.add_scalar("Huber Loss/steps", loss.item(), self._update_model_weight_step)
               
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # logging the gradients of the weights a biases of the self.dqn to tensorboard
        for name, weight_or_bias in self.dqn.named_parameters():
            self.writer.add_histogram(f'{name}.grad', weight_or_bias.grad, self._update_model_weight_step)
            grad_l2_norm = weight_or_bias.grad.data.norm(p=2).item()
            self.writer.add_scalar(f'grad_norms/{name}', grad_l2_norm, self._update_model_weight_step)
        
        # updating the target network every self.target_update_step step (EX: every 1000 stepa)
        self._update_target_network_step += 1
        if self._update_target_network_step % self.target_update_step == 0:
            print("Updateing Target Network")
            self.target_dqn.load_state_dict(self.dqn.state_dict())

        
        
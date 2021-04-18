import numpy as np
import random
import copy
from collections import namedtuple, deque

from model_ma import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

#from noise import OUNoise

device = torch.device("cpu")
# default parameters
defparameters = {
    # roll over the state to next time step
    'actor_units': [256, 128],    
    # roll over the state to next time step
    'critic_units': [256, 128],         
    # roll over the state to next time step
    'lr_actor': 1e-3,    
    # learning rate of the Critic
    'lr_critic': 1e-3,                  
    
    # discount factor
    'gamma': 0.99,                      
    
    # for soft update of target parameters
    'tau': 0.15, 
    # update parameters per this number
    'update_every': 1,                  
    
    # update parameters per this number
    'seed': 42,  
    # update parameters per this number
    'weight_decay': 0,                  
    
    # Ornstein-Uhlenbeck process
    'noise_theta': 0.15,                
    'noise_sigma': 0.01,                
}

class Agent():
    """Interacts with and learns from the environment."""
    critic_local = None
    critic_target = None
    critic_optimizer = None
    
    def __init__(self, state_size, action_size, memory, device ='cpu', params = defparameters):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            memory (obj): Memory buffer to sample
            device (str): device string between cuda:0 and cpu
            params (dict): hyper-parameters
        """
        print('Agent')
        #print('== parameter ==')
        for k, v in params.items():
            print('      ',k, v)    
        print("==========")
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.step_t = 0
        self.update_every = params['update_every']

        # Set parameters
        self.gamma = params['gamma']
        self.tau = params['tau']
        self.seed = random.seed(params['seed'])

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, params['seed'],
                                 params['actor_units'][0], params['actor_units'][1]).to(device)
        self.actor_target = Actor(state_size, action_size, params['seed'],
                                  params['actor_units'][0], params['actor_units'][1]).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=params['lr_actor'])

        # Critic Networks - one for all actor
        if not Agent.critic_local:
            Agent.critic_local = Critic(state_size, action_size, params['seed'],
                                              params['critic_units'][0], params['critic_units'][1]).to(device)
        if not Agent.critic_target:
            Agent.critic_target = Critic(state_size, action_size, params['seed'],
                                               params['critic_units'][0], params['critic_units'][1]).to(device)
        if not Agent.critic_optimizer:
            Agent.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                                     lr=params['lr_critic'], weight_decay=params['weight_decay'])

        self.critic_local = Agent.critic_local
        self.critic_target = Agent.critic_target
        self.critic_optimizer = Agent.critic_optimizer

        # Noise process
        self.noise = OUNoise(action_size, params['seed'], theta=params['noise_theta'], sigma=params['noise_sigma'])

        # Replay memory
        self.memory = memory

    def store_actor_weights(self, filename):
        """Store weights of Actor

        Params
        ======
            filename (str): string of filename to store weights of actor
        """
        torch.save(self.actor_local.state_dict(), filename)

    def store_critic_weights(self, filename):
        """Store weights of Critic

        Params
        ======
            filename (str): string of filename to store weights of critic
        """
        torch.save(self.critic_local.state_dict(), filename)

    def load_actor_weights(self, filename):
        """Load weights of Actor

        Params
        ======
            filename (str): string of filename to load weights of actor
        """
        self.actor_local.load_state_dict(torch.load(filename))

    def load_critic_weights(self, filename):
        """Load weights of Critic

        Params
        ======
            filename (str): string of filename to load weights of critic
        """
        self.critic_local.load_state_dict(torch.load(filename))

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        self.step_t = (self.step_t + 1) % self.update_every

        # Learn, if enough samples are available in memory
        if self.step_t == 0 and len(self.memory) > self.memory.get_batch_size():
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        #state = torch.from_numpy(state).float().to(self.device)
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
            
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)
  
    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
#============


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (str): cuda:0 for GPU or cpu
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        
        experiences = random.sample(self.memory, k=self.batch_size)
        experiences = [e for e in experiences if e is not None]

        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)

        return states, actions, rewards, next_states, dones
        
        """Randomly sample a batch of experiences from memory."""
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)
        """
    
    
    def get_batch_size(self):
        return self.batch_size

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

#=============
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state            
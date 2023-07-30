import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED=47

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)

# PPO Actor-Critic
class Actor_PPO(nn.Module):
    def __init__(self,state_size, action_size, size1=64*4,size2=64*4,size3=32*4, seed=SEED):
        super(Actor_PPO, self).__init__()
        self.seed=torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.dense1 = nn.Linear(state_size, size1)       
        self.dense2=nn.Linear(size1, size2)        
        self.dense3=nn.Linear(size2, size3)
        self.linear_mu=nn.Linear(size3, action_size)
        self.linear_log_sigma=nn.Parameter(torch.zeros(action_size))
    
    def reset_parameters(self):
        self.dense1.weight.data.uniform_(*hidden_init(self.dense1))
        self.dense2.weight.data.uniform_(*hidden_init(self.dense2))
        self.dense3.weight.data.uniform_(*hidden_init(self.dense3))
        self.linear_mu.weight.data.uniform_(-3e-3, 3e-3)
        
        
    def forward(self, x):
        mu,sigma = self.get_params(x)
        distribution = torch.distributions.normal.Normal(mu,sigma)
        self.mu = mu
        self.sigma=sigma
        action = distribution.sample()
        return action, distribution
    
    def get_params(self, x_state):
        xs=self.dense1(x_state)
        xs=F.relu(xs)
        x=self.dense2(xs)
        x=F.relu(x)
        x=self.dense3(x)
        x=F.relu(x)
        mu=F.tanh(self.linear_mu(x))
        log_sigma = F.tanh(self.linear_log_sigma)
        sigma = (torch.exp(log_sigma)-0.3)/2 # reduce sigma range
        return(mu,sigma)
    
class Critic_PPO(nn.Module):
    def __init__(self,state_size,size1=256,size2=128*2,size3=64*2, seed=SEED):
        super(Critic_PPO, self).__init__()
        self.seed=torch.manual_seed(seed)
        self.state_size = state_size
        self.dense1=nn.Linear(state_size, size1,bias=False)
        self.dense2=nn.Linear(size1, size2,bias=False)
        self.dense3=nn.Linear(size2, size3,bias=False)
        self.dense4=nn.Linear(size3, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.dense1.weight.data.uniform_(*hidden_init(self.dense1))
        self.dense2.weight.data.uniform_(*hidden_init(self.dense2))
        self.dense3.weight.data.uniform_(*hidden_init(self.dense3))
        self.dense4.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
        x = self.dense1(state)
        x = F.leaky_relu(x)
        x = self.dense2(x)
        x = F.leaky_relu(x)
        x = self.dense3(x)
        x = F.leaky_relu(x)
        x = self.dense4(x)
        return x

    
    
    
    
    
# DDPG Actor-Critic with BatchNormalization    
class Actor(nn.Module):
    def __init__(self,state_size, action_size, size1=256,size2=128, seed=SEED):
        super(Actor, self).__init__()
        self.seed=torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.dense1 = nn.Linear(state_size, size1, bias=False)
        self.bn1 = nn.BatchNorm1d(size1,momentum=MOMENTUM,track_running_stats=TRACK_STATS, affine=AFFINE)
        self.dense2=nn.Linear(size1, size2,bias=False)
        self.bn2 = nn.BatchNorm1d(size2,momentum=MOMENTUM, track_running_stats=TRACK_STATS, affine=AFFINE)
        self.dense3=nn.Linear(size2, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.dense1.weight.data.uniform_(*hidden_init(self.dense1))
        self.dense2.weight.data.uniform_(*hidden_init(self.dense2))
        self.dense3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, x):
        x=self.dense1(x)
        x=self.bn1(x)
        x=F.relu(x)
        x=self.bn2(self.dense2(x))
        x=F.relu(x)
        x=F.tanh(self.dense3(x))
        return x
    
    
class Critic(nn.Module):
    def __init__(self,state_size,action_size, size1=256,size2=128*2,size3=64*2, seed=SEED):
        super(Critic, self).__init__()
        self.seed=torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.dense1=nn.Linear(state_size, size1,bias=False)
        self.bn1 = nn.BatchNorm1d(size1,momentum=MOMENTUM, track_running_stats=TRACK_STATS, affine=AFFINE)
        self.dense2=nn.Linear(size1+action_size, size2,bias=False)
        self.bn2 = nn.BatchNorm1d(size2,momentum=MOMENTUM, track_running_stats=TRACK_STATS, affine=AFFINE)
        self.dense3=nn.Linear(size2, size3,bias=False)
        self.bn3 = nn.BatchNorm1d(size3,momentum=MOMENTUM, track_running_stats=TRACK_STATS, affine=AFFINE)
        self.dense4=nn.Linear(size3, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.dense1.weight.data.uniform_(*hidden_init(self.dense1))
        self.dense2.weight.data.uniform_(*hidden_init(self.dense2))
        self.dense3.weight.data.uniform_(*hidden_init(self.dense3))
        self.dense3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state, action):
        xs=self.dense1(state)
        xs=self.bn1(xs)
        xs = F.leaky_relu(xs)
        x = torch.cat((xs, action), dim=1)
        x = self.dense2(x)
        x=self.bn2(x)
        x=F.leaky_relu(x)
        x = self.dense3(x)
        x=self.bn3(x)
        x = F.leaky_relu(x)
        x=self.dense4(x)
        return x
    
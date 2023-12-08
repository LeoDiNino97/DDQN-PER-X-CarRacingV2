import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#-----Double Deep Q-Learning with Prioritized Experience Replay (DDQN-PER)------

#_______________________________________________________________________________
#____________________FRAMES PREPROCESSING METHODS_______________________________
#_______________________________________________________________________________

'''
We want to do learn-from-pixels, so we have to define some processing methods
Infact we have to stack a certain number of frames to infuse markovianity into the learning process
In particular we can consider the idea of stacking raw pixels from last 4 frames as we discussed during lectures

So the idea is the following
    - PREPROCESS THE IMAGE in order to get a single channel image for each frame of a certain shape, which in case of CarRacing environment is (96,96)
    - STACK 4 subsequent frames

In this way we define a multi channel input for our network that is going to be of shape (4,84,84) (channels first standard)
'''

# Simple rescaling and channel compressing

def preprocess(image):
    # Cutting the image to exclude the stats-bar, likely a confounder for the image and to square it
    image = image[:84, 6:90]

    # Normalizing and grey scaling
    image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    return torch.tensor(image, dtype=torch.float64)

# Now time to implement a function to perform a rolling stacking of the frames while running through the main training loop 
 
def stacking(new_frame, frames):
    return torch.tensor(torch.cat((frames[1:,:,:,], new_frame.unsqueeze(0)), axis=0), dtype=torch.float32)
#_______________________________________________________________________________
#_______________________DEEP LEARNING CLASSES___________________________________
#_______________________________________________________________________________

'''
We now have to design a convolutional neural network to perform learn-from-pixels using as input (4,84,84) tensors
'''

# Firstly lets define a CNN since we want to learn from pixels to solve CarRacing-v2 environment

class CNN_Learner(nn.Module):
    def __init__(self, n_actions, stacking_size=4):
        super(CNN_Learner, self).__init__()

        # Usual thumb rule when stacking convolutional layers: double the number of filters, zooming their dimension
        # The number of input channels is the number of stacked subsequent frames

        self.conv1 = nn.Conv2d(stacking_size, 16, kernel_size=(8,8), stride=4)  
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(4,4), stride=2)             
        
        # Fully connected section of the newtork
        self.fc1 = nn.Linear(2592, 256)                                         # The input of the linear block is a flattened (32,9,9) tensor
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        # Convolutional block with rectified linear-unit activation
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        # Flatten the tensor into a vector
        feats = torch.prod(torch.tensor(x.shape[1:]))
        x = x.view((-1, feats))

        # Fully connected section
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

#_______________________________________________________________________________
#_____________________________DDQN-PER CLASSES__________________________________
#_______________________________________________________________________________

# Now let's build the experience buffer for the DDQN-PER implementation

class ReplayBuffer(nn.Module):
    def __init__(self, memory, alpha=0.5, beta=0.5, corr=0.1):
        super(ReplayBuffer).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.memory = memory        # Room in the buffer
        self.alpha = alpha          # Prioritization parameter
        self.beta = beta            # Bias-annealing parameter

        # Preallocating the buffer as different torch tensors
        self.state0_buffer = torch.zeros((memory,4,84,84)).to(self.device)
        self.actions_buffer = torch.torch.zeros(memory, dtype=torch.int32).to(self.device)
        self.rewards_buffer = torch.zeros(memory).to(self.device)
        self.done_buffer = torch.zeros(memory).to(self.device)
        self.state1_buffer = torch.zeros((memory,4,84,84)).to(self.device)

        # Preallocating priorities
        self.priorities = torch.ones(memory).to(self.device)

        # Some control variables for the buffer
        self.pos = 0
        self.size = 0

        # Correction term to avoid zero values when updating priorities
        self.corr = corr

    def buff(self, s0, a, R, done, s1):
        transition = (s0, a, R, done, s1)
        self.state0_buffer[self.pos,:,:,:] = transition[0].clone().detach()
        self.actions_buffer[self.pos] = torch.tensor(transition[1],dtype=torch.int32)
        self.rewards_buffer[self.pos] = torch.tensor(transition[2])
        self.done_buffer[self.pos] = torch.tensor(transition[3])
        self.state1_buffer[self.pos,:,:,:] = transition[4].clone().detach()

        # SARS experiences are stored with maximum priority 
        self.priorities[self.pos] = torch.max(self.priorities[:(self.pos+1)]).item()

        # Counter for the storing position, eventually reinitializing if full 
        self.pos = (self.pos+1) % self.memory
        self.size = min(self.size+1, self.memory)

    def sample(self, batch_size=32):

        # Sample a batch of buffered SARS experiences using as probability rules the normalized priorities 
        # (Most similar implementation of np.random.choice in torch uses multinomial method for a tensor as a distribution of probability in multiple samples )

        P=self.priorities[:self.size]**self.alpha/torch.sum(self.priorities[:self.size]**self.alpha)
        sampled = P.multinomial(batch_size, replacement=False)

        sampled_S0 = self.state0_buffer[sampled,:,:,:]
        sampled_A = self.actions_buffer[sampled]
        sampled_R = self.rewards_buffer[sampled]
        sampled_D = self.done_buffer[sampled]
        sampled_S1 = self.state1_buffer[sampled,:,:,:]
        sampled_SARS = (sampled_S0, sampled_A, sampled_R, sampled_D, sampled_S1)

        # Compute normalized importance sampling weights as bias correction strategy
        weights = torch.mul(self.size, self.priorities)[sampled]**(-self.beta)
        weights = weights/torch.max(weights)

        return sampled, sampled_SARS, weights

    def update_priority(self, indices, errors):
        
        # Update priorities based on sampled batch 
        self.priorities[indices] = torch.abs(errors) + self.corr

# Finally let's define the DDQN-PER agent

class Policy:
    def __init__(self, 
                 batch_size=128, 
                 stepsize=0.00025, 
                 replay_period=500,
                 warm_up=1000, 
                 net_sync=10000, 
                 budget=1e6, 
                 epsilon=1, 
                 memory=20000, 
                 alpha=0.5, 
                 beta=0.5, 
                 gamma=0.99):
        
        self.env = gym.make('CarRacing-v2', continuous=False)
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

        self.continuous = False
        self.state_dim = (4,84,84)
        self.action_dim = self.env.action_space.n

        #________________________________________________________________
        #____________DEFINING THE INITIAL OBSERVATION STATE______________
        s, _ = self.env.reset()

        # Skip the initial zooming animation
        for i in range(50):
            s, _, _, _, _ = self.env.step(0)
        
        # Initial state is a tiling of this initial frame repeated 4 times
        self.s0 = torch.tensor(np.tile(preprocess(s), (4, 1, 1)).astype(np.float32))
        self.s0 = self.s0.to(self.device)
        #_________________________________________________________________
        #_________________________________________________________________

        self.replay_period = replay_period                          # Clock for the replay phase
        self.budget = budget                                        # Maximum number of steps
        self.net_sync = net_sync                                    # Clock for the transfer of parameters between current net and target net

        self.warm_up = warm_up                                      # Number of iterations before starting epsilon decay
        self.epsilon_min = 0.1                                      # Minimum value reachable for epsilon decay

        self.net = CNN_Learner(n_actions=self.action_dim)
        self.net = self.net.to(self.device)
        self.target_net = CNN_Learner(n_actions=self.action_dim).to(self.device)
        self.target_net = self.target_net.to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())      # Same initialization for the two networks

        self.batch_size = batch_size
        self.buffer = ReplayBuffer(memory, alpha, beta)
        
        self.step_count = 0

        self.gamma = gamma
        self.epsilon = epsilon
        self.reward = 0

        self.stepsize = stepsize
        self.optimizer = torch.optim.RMSprop(self.net.parameters(),
                                             lr=self.stepsize)
        
        # Inner parameters for act evaluation

        self.last_action = 0
        self.metronome = 4
        self.counter = 0
        self.actions = {0:"Nothing", 1:"Steer left", 2:"Steer right", 3:"Gas", 4:"Brake"}
        self.first = True

    def forward(self):
        with torch.no_grad():
            # Epsilon-greedy approach 
            p = np.random.uniform(low=0, high=1)
            if p >= self.epsilon: 
                # Be greedy in the policy
                action = int(torch.argmax(self.net(self.s0.unsqueeze(0))))
            else:
                # Explore!
                action = int(self.env.action_space.sample())

            # Simulate action, but remember: we are stacking 4 subsequent frames to inject markovianity
            # Standard approach is repeating this action over the 4 stacked frames

            R = 0
            for _ in range(4):
                s_, r, terminated, truncated, _ = self.env.step(action)
                self.reward += r
                R += r
                s_ = preprocess(s_)
                s_ = s_.to(self.device)
                s1 = stacking(s_,self.s0)  
                if terminated or truncated:
                    break

            if (self.reward >= 0) and (self.step_count > self.warm_up): 
                # Epsilon linear decay
                if self.epsilon > self.epsilon_min:
                    self.epsilon -= 0.005
                
            done = terminated or truncated
            
            # Put experience in the buffer
            self.buffer.buff(self.s0, action, R, done, s1)

            # Update the state of the system
            self.s0 = s1
            self.s0 = self.s0.to(self.device)
            self.step_count += 1

            # Env-restart subloop 
            if done:
                s, _ = self.env.reset()
                for i in range(50):
                    s, _, _, _, _ = self.env.step(0)
                self.s0 = torch.tensor(np.tile(preprocess(s), (4, 1, 1)).astype(np.float32))
                self.s0 = self.s0.to(self.device)
                self.reward = 0
    
    def train(self):
        while self.step_count < self.budget: 
            self.forward()
            if self.step_count % self.replay_period == 0:
                torch.autograd.set_detect_anomaly(True)
                #________________________REPLAY PHASE STARTS________________________

                print('The model is now learning from previous experiences!')

                sampled, sampled_SARS, weights = self.buffer.sample(self.batch_size)
                weights = weights.detach()

                # Compute TD error
                # SARS = (s0, a, r, done, s1)

                s0_ = sampled_SARS[0]
                s1_ = sampled_SARS[4]
                
                target_Qj1 = self.target_net(s1_).to(self.device)
                Qj1 = self.net(s1_).to(self.device)
                Qj0 = self.net(s0_).to(self.device)

                target = (sampled_SARS[2]                                                                                # Reward
                          + self.gamma*target_Qj1[torch.arange(len(target_Qj1)),torch.argmax(Qj1,axis=1)])               # Target evaluation on current selection
                
                current =  Qj0[torch.arange(len(Qj0)),sampled_SARS[1]]                                                   # Current evaluation

                target = target.to(self.device)
                current = current.to(self.device)

                TD_err = target-current 

                # Update priorities
                self.buffer.update_priority(sampled, TD_err)

                # Backward step for net optimization
                # Backpropagate the loss to compute gradients

                # Define the loss as a weighted average of the squared temporal difference errors
                loss = torch.sum(weights*(target-current)**2)/torch.sum(weights)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                #_________________________REPLAY PHASE ENDS___________________________
            
            # Transfer net parameters to target net on sync clock
            if self.step_count % self.net_sync == 0:
                self.target_net.load_state_dict(self.net.state_dict())

    def act(self,s): 

        # This method must reflect every design choice we made in the implementation 

        if self.first == True:
            # Skipping the introduction also in the evaluation 
            while self.counter < 50:
                self.counter += 1
                return 0
            
            # Once it is called the first time the behaviour changes
            self.first == False
            self.last_action = int(torch.argmax(self.net(self.s0.unsqueeze(0))))
        else:
            # Once every 4 frames it chooses a new action
            if (self.counter % self.metronome) == 0:
                self.s0 = stacking(preprocess(s).to(self.device),self.s0).to(self.device)
                self.last_action = int(torch.argmax(self.net(self.s0.unsqueeze(0))))

        self.s0 = stacking(preprocess(s).to(self.device),self.s0).to(self.device)
        return self.last_action

    def save(self):
        torch.save(self.net.state_dict(), 'model.pt')

    def load(self):
        self.net.load_state_dict(torch.load('model.pt'))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret

#-------------------------------------------------------------------------------
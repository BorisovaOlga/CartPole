'''
I use these video tutorial
https://youtube.com/playlist?list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi&si=ivEXswCHP_GqJnZ-

Files on GitHub
https://github.com/johnnycode8/dqn_pytorch
'''



import gymnasium as gym
import yaml # import hyperparameters
import matplotlib 
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import random
import numpy as np

from dqn import DQN
from replay_memory import ReplayMempry



DATE_FORMAT = "%m-%d %H:%M:%S"

RUNS_DIR = 'runs'
os.makedirs(RUNS_DIR, exist_ok=True) # create directory

matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'



class Agent:
    # add hyperparameters set
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set
        
        # Hyperparameters
        self.replay_memory_size = hyperparameters['replay_memory_size'] # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']    # size of the trainig data set sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']       # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']      # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']        # minimum epsilon value
        self.learning_rate_a    = hyperparameters['learning_rate_a']
        self.discount_factor_g  = hyperparameters['discount_factor_g']
        self.network_sync_rate       = hyperparameters['network_sync_rate']  
        self.env_id             = hyperparameters['env_id']
        self.stop_on_reward     = hyperparameters['stop_on_reward']         # stop training after reaching this number of rewards
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.env_make_params    = hyperparameters.get('env_make_params',{}) # Get optional environment-specific parameters, default to empty dict


        # Neural Network
        self.loss_fn = nn.MSELoss()                                     # NN Loss function. MSE=Mean Squared Error can be swapped to something else
        self.optimizer = None                                           # NN Optimizer

        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def run(self, is_training = True, render=False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        # Create instance of the environment.
        # Use "**self.env_make_params" to pass in environment-specific parameters from hyperparameters.yml.
        env = gym.make('CartPole-v1', render_mode='human' if render else None, **self.env_make_params)

        # POLICY
        num_states = env.observation_space.shape[0] # Get observation space size
        num_actions = env.action_space.n            # Number of possible actions
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device) # Create policy and target network. Number of nodes in the hidden layer can be adjusted

        reward_per_episode = [] # add a variable to keep track the reward per episode 
        epsilon_history = []

        if is_training:
            # initialize replay memory
            memory = ReplayMempry(self.replay_memory_size)

            # epsilon-greedy algorithm
            epsilon = self.epsilon_init # Initialize epsilon
            
            # Create Target Network and make it identical to the policy network
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # Policy network optimizer
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            # Track number of steps taken. Used for syncing policy => target network
            step_count = 0

            # Track best reward
            best_reward = -9999999
        else:
            # Load learned policy
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

            # switch model to evaluation mode
            policy_dqn.eval()

        # TRAINING

        for episode in range(1000):
            state, _ = env.reset()                                        # at the beginig of the episode the environment will reset 
            state = torch.tensor(state, dtype=torch.float, device=device) # Convert state into tensor
            
            terminated = False      # True when agent reaches goal or fails
            episode_reward = 0.0    # Used to accumulate rewards per episode

            # Perform actions until episode terminates or reaches max rewards
            # (on some envs, it is possible for the agent to train to a point where it NEVER terminates, so stop on reward is necessary)
            while (not terminated and episode_reward < self.stop_on_reward):

                if is_training and random.random() < epsilon: # If we are training and we generate a random number (0,1) 
                                                              # AND if this number is less then epsilon, which start 100% random - 
                                                              # we will do random action.
                    action = env.action_space.sample()        # Next action. Feed the observation to agent
                    action = torch.tensor(action, dtype=torch.int64, device=device) # Convert action to tensor

                else:                                         # Othewise we will select action that policy network prescribes
                    with torch.no_grad():                     # Auto calculation gradient during training
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()   # Pass state to the policy. It gives Q-value. 
                                                                               # We want to get index of the hiest number - use argmax.
                                                                               # Function unsqueeze(dim=0) adds extra dimension at the begining. 
                                                                               # This is needed for network (2-dimentional data)

                # Processing
                new_state, reward, terminated, _, info = env.step(action.item()) # function item() is needed to get the tensor value 

                # Acumulate reward
                episode_reward += reward

                # Convert new state and reward to tensors
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)


                if is_training:
                    memory.append((state, action, new_state, reward, terminated))

                    # Increment step count. Used for syncing policy => target network
                    step_count += 1

                # Move to new state
                state = new_state

            reward_per_episode.append(episode_reward) # add new reward


            # Save model when new best reward is obtained.
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

            # Update graph every x seconds
            current_time = datetime.now()
            if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(reward_per_episode, epsilon_history)
                    last_graph_update_time = current_time
            

            # If enough experience has been collected
            if len(memory) > self.mini_batch_size:
                # Sample from memory
                mini_batch = memory.sample(self.mini_batch_size) # We call memory sample. Pass it batch size yo get batch

                self.optimize(mini_batch, policy_dqn, target_dqn) # Optinization function

                # After one episode do epsilon smaller
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0


    # PLOTS

    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)



    # Optimization function
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        
        # Transpose the list of experiences and aeparate each element
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create  batch tensors
        # tensor([[1,2,3]])
        states       = torch.stack(states)
        actions      = torch.stack(actions)
        new_states   = torch.stack(new_states)
        rewards      = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            # Calculate target Q-values
            target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
            ''' 
            target_dqn(neq_states) ==> tensor([[1,2,3],[4,5,6]])
                .max(dim=1)        ==> torch.return_types.max(values=tensor([3,6]), indices=tensor([3,0,0,1]))
                    [0]            ==> tensor([3,6])
            '''


        # Calculate Q-values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
      

        # Compute loss for the whole mini batch
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad() # Clear gradient
        loss.backward()            # Compute gradients (backpropagation)
        self.optimizer.step()      # Update network parameters i.e. weights and biases 




if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)
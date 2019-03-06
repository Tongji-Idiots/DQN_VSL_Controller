# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 13:36:09 2019

@author: ChocolateDave
"""

# Import Modules
import collections
import numpy as np
import os,sys
sys.path.append("./lib")
sys.path.append("./common")
import torch
import torch.nn as nn
import torch.optim as optim

import env as Env
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from common import action, agent, utils, experience, tracker

# Global Variable:
#parser.add_argument("--resume", default = None, type = str, metavar= path, help= 'path to latest checkpoint')
params = utils.Constants

# Build Up Neural Network
class DuelingNetwork(nn.Module):
    """
    Create a neural network to convert image data
    """
    def __init__(self, input_shape, n_actions):
        super(DuelingNetwork, self).__init__()

        self.convolutional_Layer = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fully_connected_adv = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.fully_connected_val = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        #print('1, *shape: ', torch.zeros(1, *shape).size())
        o = self.convolutional_Layer(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 100
        conv_out = self.convolutional_Layer(fx).view(fx.size()[0], -1)
        val = self.fully_connected_val(conv_out)
        adv = self.fully_connected_adv(conv_out)
        return val + adv - adv.mean()

class DQN(nn.Module):
    """Basic neural network framework"""
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=9, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 100
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)

# Saving model
def save_model(net, buffer, beta, optim, path_net, path_buffer, frame):
	torch.save({
		'frame': frame,
		'state_dict': net.state_dict(),
        #optimizer:
		'optimizer': optim},
		path_net)
	'''torch.save({
        #prioritized replay params:
        'buffer': buffer.buffer,
        'priorities': buffer.priorities,
        'pos': buffer.pos},
        path_buffer)'''

# Load pretrained model
def load_model(net, path_net, path_buffer):
	state_dict = torch.load(path_net)
	net.load_state_dict(state_dict['state_dict'])
	frame = state_dict['frame']
	optimizer = state_dict['optimizer']
	print("Having previously run %d frames." % frame)
	'''buffer_dict = torch.load(path_buffer)
	buffer = buffer_dict['buffer']
	priorities = buffer_dict['priorities']
	pos = buffer_dict['pos']'''
	net.train()
	return net, frame + 1, optimizer

# Training
def Train():   
    writer = SummaryWriter(comment = '-VSL-Dueling')
    env = Env.SumoEnv(frameskip= 15)
    env.unwrapped
    net = DuelingNetwork(env.observation_space.shape, env.action_space.n)

    path_net = os.path.join('./savednetwork/', 'network_checkpoint.pth')
    path_buffer = os.path.join('./savednetwork/', 'buffer_checkpoint.pth')
    print("CUDA™ is " + ("AVAILABLE" if torch.cuda.is_available() else "NOT AVAILABLE"))
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device("cuda")
        net.to(device)
        torch.backends.cudnn.benchmark = True
        if next(net.parameters()).is_cuda:
            print("Now using {} for training".format(torch.cuda.get_device_name(torch.cuda.current_device())))
    else:
        device = torch.device("cpu")
        print("Now using CPU for training")
    
    print("Observation space: {}, Action size:{}".format(env.observation_space.shape,env.action_space.n))

    tgt_net = agent.TargetNet(net)
    selector = action.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = tracker.EpsilonTracker(selector, params)
    agents = agent.DQNAgent(net, selector, device = device)

    exp_source = experience.ExperienceSourceFirstLast(env, agents, gamma=params['gamma'], steps_count=1)
    #buffer = experience.ExperienceReplayBuffer(exp_source, params['replay_size']) # For Regular memory optimization
    buffer = experience.PrioReplayBuffer(exp_source, params['replay_size'],params['PRIO_REPLAY_ALPHA']) #For Prioritized memory optimization

    frame_idx = 0
    flag = True
    beta = params['BETA_START']

    #Add graph
    if frame_idx == 0:
        print("=> Now drawing graph...")
        state = env.reset()
        state = np.expand_dims(state, 0)
        state = torch.from_numpy(state).to(device)
        writer.add_graph(net, state)
        print("=> Graph done!")
        env.close()
        del state

    #Load previous network
    if path_net and path_buffer:
        if os.path.isfile(path_net) and os.path.isfile(path_buffer):
            print("=> Loading checkpoint '{}'".format(path_net))
            net, frame_idx, optimizer = load_model(net, path_net, path_buffer)
            print("Checkpoint loaded successfully! ")
        else:
            optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])
            print("=> No such checkpoint at '{}'".format(path_net))

    with tracker.RewardTracker(writer, params['stop_reward']) as reward_tracker:  #stop reward needs to be modified according to reward function
        while True:
            frame_idx += 1
            buffer.populate(1)
            epsilon_tracker.frame(frame_idx)
            beta = min(1.0, params['BETA_START'] + frame_idx * (1.0 - params['BETA_START']) / params['BETA_FRAMES'])

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                writer.add_scalar("Interaction/Beta", beta, frame_idx)
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    env.close()
                    break

            if len(buffer) < params['replay_initial']:
                continue

            #Regular memory optimization
            '''optimizer.zero_grad()
            batch = buffer.sample(params['batch_size'])
            loss_v = utils.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params['gamma'], device=device)
            loss_v.backward()
            optimizer.step()'''

            #Prioritized memory optimization
            if flag:
                print("\nTraining begins...")
                flag = False
            optimizer.zero_grad()
            batch, batch_indices, batch_weights = buffer.sample(params['batch_size'], beta)
            loss_v, sample_prios_v= utils.calc_loss(batch, batch_weights, net, tgt_net.target_model,
                                               params['gamma'], device=device)
            loss_v.backward()
            optimizer.step()
            buffer.update_priorities(batch_indices, sample_prios_v.data.cpu().numpy())

            #Writer function -> Tensorboard file
            writer.add_scalar("Train/Loss", loss_v, frame_idx)
            for name, netparam in net.named_parameters():
                    writer.add_histogram('Model/{}'.format(name), netparam.clone().cpu().data.numpy(), frame_idx)
            
            #saving model
            if new_rewards:
                save_model(net, buffer, beta, optimizer, path_net, path_buffer, frame_idx)
                print("=> Checkpoint reached.\n=>Network saved at %s" % path_net)
            
            
            if frame_idx % params['max_tau'] == 0:
                tgt_net.sync()  #Sync q_eval and q_target

if __name__ == '__main__':
    Train()
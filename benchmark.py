import collections
import numpy as np
import os,sys
sys.path.append("./lib")
sys.path.append("./common")
import torch
import torch.nn as nn
import torch.optim as optim
import xml.etree.ElementTree as ET

import Env as benv
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from common import action, agent, utils, experience, tracker
import benchmark_draw

#Global Variables
Total_Time = 720
params = utils.Constants

# Build Up Neural Network
class DuelingNetwork(nn.Module):
    """
    Create a neural network to convert image data
    """
    def __init__(self, input_shape, n_actions):
        super(DuelingNetwork, self).__init__()

        self.convolutional_Layer = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=9, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
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

# Load pretrained model
def bench_load_model(net, path_net):
	state_dict = torch.load(path_net)
	net.load_state_dict(state_dict['state_dict'])
	frame = state_dict['frame']
	print("Having pre-trained %d frames." % frame)
	net.eval()
	return net

def custom_opt(obs) -> int:
    ms = obs["ms"]
    if ms>=0 and ms<=8:
        action = 0
    elif ms<=11.11:
        action = 1
    elif ms<=13.89:
        action = 2
    elif ms<=16.67:
        action = 3
    else:
        action = 4
    return action

def runtime(i):
    num_arrow = int(i * 50 / Total_Time)
    num_line = 50 - num_arrow
    percent = i * 100.0 / Total_Time
    process_bar = 'Scenario Running... [' + '>' * num_arrow + '-' * num_line + ']' + '%.2f' % percent + '%' + '\r'
    sys.stdout.write(process_bar)
    sys.stdout.flush()

class BenchmarkEpsilonTracker:
    def __init__(self, epsilon_greedy_selector, params):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon_final = params['epsilon_final']
        self.frame(0)

    def frame(self, frame):
        self.epsilon_greedy_selector.epsilon = self.epsilon_final

def benchmark():
    writer = SummaryWriter(comment="-benchmark")

    tree = ET.parse("benchmark.data.xml")
    root = tree.getroot()

    env = benv.SumoEnv(frameskip = 15)
    #First run a scenario without VSL
    ori = root.find("origin")
    env.reset()
    data_ori = np.empty((2, Total_Time), dtype=np.float32)
    for i in range(Total_Time):
        _, _, _, ori_dict = env.step(4)
        data_ori[0][i] = ori_dict["ms"]
        data_ori[1][i] = ori_dict["ttt"]

        new_data = ET.Element("data",{"frame":str(i),"ms":str(ori_dict["ms"]),"ttt":str(ori_dict["ttt"])})
        ori.append(new_data)
        tree.write("benchmark.data.xml")

        runtime(i)
    env.close()

    #Then run a scenario with custom VSL regulations
    custom = root.find("custom")
    data_cust = np.empty((2, Total_Time), dtype=np.float32)
    act = 4
    env.reset()
    for i in range(Total_Time):
        _, _, _, custom_dict = env.step(act)
        act = custom_opt(custom_dict)
        data_cust[0][i] = custom_dict["ms"]
        data_cust[1][i] = custom_dict["ttt"]

        new_data = ET.Element("data",{"frame":str(i),"ms":str(custom_dict["ms"]),"ttt":str(custom_dict["ttt"])})
        ori.append(new_data)
        tree.write("benchmark.data.xml")
        runtime(i)
    env.close()
    
    #Finally run a scenario with neural network guided VSL
    nn_xml = root.find("nn")
    net = DuelingNetwork(env.observation_space.shape, env.action_space.n)
    path_net = os.path.join('./savednetwork/', 'network_checkpoint.pth')
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
    if path_net:
        if os.path.isfile(path_net):
            print("=> Loading checkpoint '{}'".format(path_net))
            net = bench_load_model(net, path_net)
            print("Checkpoint loaded successfully! ")
    selector = action.EpsilonGreedyActionSelector(epsilon=params['epsilon_final'])
    epsilon_tracker = BenchmarkEpsilonTracker(selector, params)
    agents = agent.DQNAgent(net, selector, device = device)
    state = env.reset()
    data_nn = np.empty((2, Total_Time), dtype=np.float32)
    for i in range(Total_Time):
        act, _ = agents(state)
        state, _, _, net_dict = env.step(act)
        epsilon_tracker.frame(i)
        data_nn[0][i] = net_dict["ms"]
        data_nn[1][i] = net_dict["ttt"]
        new_data = ET.Element("data",{"frame":str(i),"ms":str(custom_dict["ms"]),"ttt":str(custom_dict["ttt"])})
        nn_xml.append(new_data)
        tree.write("benchmark.data.xml")
        runtime(i)
    env.close()

    for i in range(Total_Time):
        writer.add_scalars("Benchmark/Merging speed", {"Without VSL": data_ori[0][i], "With custom regulation \
            VSL": data_cust[0][i], "With neural network VSL": data_nn[0][i]})
        writer.add_scalars("Benchmark/Total Travel Time", {"Without VSL": data_ori[1][i], "With custom regulation \
            VSL": data_cust[1][i], "With neural network VSL": data_nn[1][i]})

if __name__ == "__main__":
    benchmark()
    benchmark_draw.draw()
        
        
    




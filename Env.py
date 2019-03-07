import numpy as np
import os,sys
sys.path.append("./lib")
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'],'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import xml.etree.ElementTree as ET

import cmath
import gym
import torch
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from collections import deque
import traci
from sumolib import checkBinary

#Environment Constants
STATE_SHAPE = (81, 441, 1)      
WARM_UP_TIME = 3 * 1e2
TOTAL_TIME = 9 * 1e3
VEHICLE_MEAN_LENGTH = 5
MAX_LENGTH = 2196.82
speeds = [11.11, 13.89, 16.67, 19.44, 22.22]  # possible actions collection

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not belive how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

class SumoEnv(gym.Env):
    '''Sumo Environment is a simulation environment which provides necessary parameters for training. On-ramp simulation
    environment could be modified in xml files in project.'''
    #Memory Organization
    __slots__ = ['frameskip', 'run_step', 'lane_list', 'vehicle_list', 'vehicle_position', \
        'lanearea_dec_list', 'lanearea_max_speed','lanearea_ob', 'action_set', 'evaluation'\
            'sumoBinary', 'projectFile', 'observation_space', 'action_space', 'maxlen', 'downsample']
    def __init__(self, frameskip= 10, downsample=5, device='cpu', evaluation=False):
        #create environment

        self.frameskip = frameskip
        self.run_step = 0
        self.lane_list = list()
        self.vehicle_list = list()
        self.vehicle_position = list()
        self.lanearea_dec_list = list()
        self.lanearea_max_speed = dict()
        self.action_set = dict()
        self.downsample = downsample
        self.maxlen = 1+round((1+round(MAX_LENGTH/5))/5)
        self.evaluation = evaluation
        self.device = device
        if self.evaluation:
            self.eval_seed = self.seed()[1]

        # initialize sumo path
        self.sumoBinary = " "
        self.projectFile = './Project/'    

        # initialize lane_list and edge_list
        net_tree = ET.parse("./Project/ramp.net.xml")
        for lane in net_tree.iter("lane"):
            self.lane_list.append(lane.attrib["id"])
        self.observation_space = spaces.Box(low= -1, high=100, shape=(1, 3*len(self.lane_list), \
            self.maxlen), dtype=np.float32)

        # initialize lanearea_dec_list
        dec_tree = ET.parse("./Project/ramp.add.xml")
        for lanearea_dec in dec_tree.iter("laneAreaDetector"):
            if lanearea_dec.attrib["freq"] == '60':
                self.lanearea_dec_list.append(lanearea_dec.attrib["id"])
                self.lanearea_max_speed[lanearea_dec.attrib["id"]] = 22.22
 
        # initalize action set
        i = 0
        j = 0
        lanearea = list()
        while i < len(self.lanearea_dec_list):
            lanearea.append(self.lanearea_dec_list[i])
            if (i + 1) % 3 == 0:
                for speed in speeds:
                    self.action_set[j] = [lanearea.copy(), speed]
                    j += 1
                lanearea.clear()
            i += 1
        self.action_space = spaces.Discrete(len(self.action_set))

        # initialize vehicle_list and vehicle_position
        run_step = 0
        while run_step< TOTAL_TIME + 2:
            self.vehicle_list.append(dict())
            self.vehicle_position.append(dict())
            for lane in net_tree.iter("lane"):
                self.vehicle_list[run_step][lane.attrib["id"]]=list()
                
                self.vehicle_position[run_step][lane.attrib["id"]]=[0]*int(float(lane.attrib["length"])/VEHICLE_MEAN_LENGTH + 2)
            run_step += 1       

    def is_episode(self):
        if self.run_step == TOTAL_TIME:
            print('Scenario ends... at phase %d' % (self.run_step / 1800 + 1))
            traci.close()
            return True
        return False

    def warm_up_simulation(self):
        # Warm up simulation.
        warm_step=0
        while warm_step <= WARM_UP_TIME:
            traci.simulationStep()
            warm_step += 1
    
    def update_target_vehicle_set(self):
        # Update vehicle ids in the target area
        for lane in self.lane_list:
            self.vehicle_list[self.run_step][lane] = traci.lane.getLastStepVehicleIDs(lane)
    
    def transform_vehicle_position(self):
        # Store vehicle positions in matrices.
        for lane in self.lane_list:
            lane_shape = traci.lane.getShape(lane)
            for vehicle in self.vehicle_list[self.run_step][lane]:
                vehicle_pos= traci.vehicle.getPosition(vehicle)
                index = abs(int((vehicle_pos[0]-lane_shape[0][0])/VEHICLE_MEAN_LENGTH))
                self.vehicle_position[self.run_step][lane][index]=1
        return [self.lane_list, self.vehicle_position]
    
    def update_observation(self):

        self.update_target_vehicle_set()
        self.transform_vehicle_position()
        state = np.zeros((1, 3*len(self.lane_list), self.maxlen), dtype = np.float32)

        vehicle_position = np.zeros((len(self.lane_list),self.maxlen),dtype = np.float32)
        vehicle_speed = np.zeros((len(self.lane_list),self.maxlen),dtype = np.float32)
        vehicle_acceleration = np.zeros((len(self.lane_list),self.maxlen),dtype = np.float32)

        #originally set -1 on no road sections (abandoned)
        '''for lane in self.lane_list:
            lane_index = self.lane_list.index(lane)
            lane_len = traci.lane.getLength(lane)
            lane_stop = int (lane_len / VEHICLE_MEAN_LENGTH/self.downsample) 
            for i in range(lane_stop, self.maxlen):
                vehicle_position[lane_index][i] = -1.0'''

        current_step_vehicle = list()
        for lane in self.lane_list:
            current_step_vehicle += (self.vehicle_list[self.run_step][lane])

        for vehicle in current_step_vehicle:
            vehicle_in_lane = traci.vehicle.getLaneID(vehicle)
            lane_index = self.lane_list.index(vehicle_in_lane)
            vehicle_pos= traci.vehicle.getPosition(vehicle)
            lane_shape = traci.lane.getShape(vehicle_in_lane)
            vehicle_index = abs(int((vehicle_pos[0]-lane_shape[0][0])/VEHICLE_MEAN_LENGTH))
            vehicle_index = round(vehicle_index / self.downsample)

            vehicle_position[lane_index][vehicle_index] += 1.0
            vehicle_speed[lane_index][vehicle_index] += traci.vehicle.getSpeed(vehicle) 
            vehicle_acceleration[lane_index][vehicle_index] += traci.vehicle.getAcceleration(vehicle)

        for lane_num in range(len(self.lane_list)):
            for vehicle_num in range(len(vehicle_position[lane_num])):
                if vehicle_position[lane_num][vehicle_num] == 0 or vehicle_position[lane_num][vehicle_num] == -1:
                    continue
                vehicle_speed[lane_num][vehicle_num] /= vehicle_position[lane_num][vehicle_num]
                vehicle_acceleration[lane_num][vehicle_num] /= vehicle_position[lane_num][vehicle_num]
        
        state = np.concatenate((vehicle_position, vehicle_speed, vehicle_acceleration), axis= 0)
        return state
    
    def _getmergingspeed(self):
        ms = list()
        for lane in self.lane_list:
            if "merging" in lane:
                ms.append(traci.lane.getLastStepMeanSpeed(lane))
        meanspeed = np.mean(ms)
        return meanspeed
    
    def _gettotalwaitingtime(self):
        twt = 0.0
        for lane in self.lane_list:
            twt += traci.lane.getWaitingTime(lane)
        return twt

    def _transformedtanh(self, x, alpha=1):
        return (np.exp(x/alpha) - np.exp(-x/alpha))/(np.exp(x/alpha) + np.exp(-x/alpha))
    
    def step_reward(self):
        #Using waiting_time to present reward.
        return self._transformedtanh((self._getmergingspeed()-18)*0.8) \
             - self._transformedtanh(self._gettotalwaitingtime()*0.09)
    
    def reset_vehicle_maxspeed(self):
        for lane in self.lane_list:
            max_speed = traci.lane.getMaxSpeed(lane)
            for vehicle in self.vehicle_list[self.run_step][lane]:
                traci.vehicle.setMaxSpeed(vehicle,max_speed)
        
        for dec_lane in self.lanearea_dec_list:
            vehicle_list = traci.lanearea.getLastStepVehicleIDs(dec_lane)
            max_speed = self.lanearea_max_speed[dec_lane]
            for vehicle in vehicle_list:
                traci.vehicle.setMaxSpeed(vehicle,max_speed)

    def status(self):
        num_arrow = int(self.run_step * 50 / TOTAL_TIME)
        num_line = 50 - num_arrow
        percent = self.run_step * 100.0 / TOTAL_TIME
        process_bar = 'Scenario Running... [' + '>' * num_arrow + '-' * num_line + ']' + '%.2f' % percent + '%' + '\r'
        sys.stdout.write(process_bar)
        sys.stdout.flush()
    
    def step(self, a):
        # Conduct action, update observation and collect reward.
        reward = 0.0
        action = self.action_set[a]
        for i in range(3):
            self.lanearea_max_speed[action[0][i]]=action[1]
        if isinstance(self.frameskip, int):
            num_steps = self.frameskip
        else:
            num_steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])
        self.reset_vehicle_maxspeed()
        for _ in range(num_steps):
            traci.simulationStep()
            reward += self.step_reward()
            #self.status()
            self.run_step += 1
        # Update observation of environment state.
        observation = torch.from_numpy(self.update_observation()).unsqueeze(0).to(self.device)
        return observation, reward / num_steps, self.is_episode(), {}

    def reset(self):
        # Reset simulation with the random seed randomly selected the pool.
        if self.evaluation:
            self.sumoBinary = "sumo"
            seed = self.eval_seed
            traci.start([self.sumoBinary, '-c', self.projectFile + 'ramp.sumo.cfg', '--start','--seed',\
                 str(seed), '--quit-on-end'], label='evaluation')
            self.scenario = traci.getConnection('evaluation')
        else:
            self.sumoBinary = "sumo"
            seed = self.seed()[1]
            traci.start([self.sumoBinary, '-c', self.projectFile + 'ramp.sumo.cfg', '--start','--seed',\
                 str(seed), '--quit-on-end'], label='training')
            self.scenario = traci.getConnection('training')

        self.warm_up_simulation()

        self.run_step = 0

        obs = torch.from_numpy(self.update_observation()).unsqueeze(0).to(self.device)

        return obs 
    
    def seed(self, seed= None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        return [seed1, seed2]
    
    def close(self):
        traci.close()
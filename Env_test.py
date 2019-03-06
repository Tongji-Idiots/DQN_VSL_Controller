import numpy as np
import os,sys
sys.path.append("./lib")
import xml.etree.ElementTree as ET

import cmath
import gym
import traci
import torch
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from collections import deque
from sumolib import checkBinary
import matplotlib.pyplot as plt
def seed(seed= None):
    np_random, seed1 = seeding.np_random(seed)
    # Derive a random seed. This gets passed as a uint, but gets
    # checked as an int elsewhere, so we need to keep it below
    # 2**31.
    seed2 = seeding.hash_seed(seed1 + 1) % 2**31
    return [seed1, seed2]

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'],'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
projectFile = './project/'
sumoBinary = "sumo"
seed = seed()[1]
traci.start([sumoBinary, '-c', projectFile + 'ramp.sumo.cfg', '--start','--seed', str(seed), '--quit-on-end'], label='training')
scenario = traci.getConnection('training')

# initialize lane_list and edge_list
lane_list = list()
lane_length = list()
lanearea_dec_list = list()
lanearea_max_speed = dict()
lanearea_ob = list()
variable_speed_sign = list()
net_tree = ET.parse("./project/ramp.net.xml")
for lane in net_tree.iter("lane"):
    lane_list.append(lane.attrib["id"])
    lane_length.append(float(lane.attrib["length"]))

# initialize lanearea_dec_list
dec_tree = ET.parse("./project/ramp.add.xml")
for lanearea_dec in dec_tree.iter("laneAreaDetector"):
    if lanearea_dec.attrib["freq"] == '60':
        lanearea_dec_list.append(lanearea_dec.attrib["id"])
        lanearea_max_speed[lanearea_dec.attrib["id"]] = 22.22
    else:
        lanearea_ob.append(lanearea_dec.attrib["id"])
for sign in dec_tree.iter("variableSpeedSign"):
    variable_speed_sign.append(sign.attrib["id"])

def warm_up_simulation():
    # Warm up simulation.
    warm_step=0
    while warm_step <= 1 * 1e2:
        traci.simulationStep()
        warm_step += 1

def _getmaxspeed():
    global lane_list
    v = list()
    for lane in lane_list:
        vehicle_l = traci.lane.getLastStepVehicleIDs(lane)
        for vehicle in vehicle_l:
            v.append(traci.vehicle.getMaxSpeed(vehicle))
    return np.max(v)
    
def _getsaturation():
    global lane_list
    saturation = list()
    for lane in lane_list:
        if "merging" in lane:
            saturation.append(traci.lane.getLastStepVehicleNumber(lane)/(traci.lane.getLength(lane) / 5))
    ans = np.mean(saturation)
    #print("saturation:" + str(ans))
    saturation = ans
    return saturation

def _getmergingspeed():
    global lane_list
    ms = list()
    for lane in lane_list:
        if "merging" in lane:
            ms.append(traci.lane.getLastStepMeanSpeed(lane))
    meanspeed = np.mean(ms)
    return meanspeed
    
def _gettotalwaitingtime():
    global lane_list
    twt = 0.0
    for lane in lane_list:
        twt += traci.lane.getWaitingTime(lane)
    return twt

def reset_vehicle_maxspeed(vsl):
    for dec_lane in lanearea_dec_list:
        vehicle_list = traci.lanearea.getLastStepVehicleIDs(dec_lane)
        max_speed = vsl
        for vehicle in vehicle_list:
            traci.vehicle.setMaxSpeed(vehicle,max_speed)

def _transformedtanh(x, alpha=1):
        return (np.exp(x/alpha) - np.exp(-x/alpha))/(np.exp(x/alpha) + np.exp(-x/alpha))

def step_reward(x,y):
    #Using waiting_time to present reward.
    return (_transformedtanh((x -12)*0.4) \
        - _transformedtanh((y -27)*0.4)) / 2

def Vopt(ttt):
    action_set = [10.31, 11.35, 12.39, 13.42, 14.46]
    if ttt>=0:
        pass
    pass

twt_list = list()
merging_speed = list()
warm_up_simulation()
fig = plt.figure(figsize = (12, 8))
for i in range(10800):
    traci.simulationStep()
    ms = _getmergingspeed()
    sat = _getsaturation()
    twt = _gettotalwaitingtime()
    #print(_getmaxspeed())
    twt_list.append(twt)
    merging_speed.append(ms)
    num_arrow = int(i * 50 / 10800)
    num_line = 50 - num_arrow
    percent = i * 100.0 / 10800
    process_bar = 'Scenario Running... [' + '>' * num_arrow + '-' * num_line + ']' + '%.2f' % percent + '%' + '\r'
    sys.stdout.write(process_bar)
    sys.stdout.flush()
    #print("Steps: %d" % i, "Meanspeed: %.2f" % ms, " Saturation: %.4f" % sat, " Total travel time: %.4f" % ttt)
print("max waiting time: %.4f" % np.max(twt_list), " 15 percent travel time: %.4f" % np.percentile(twt_list, 15), \
    " 85 percent merging speed: %.2f" % np.percentile(merging_speed, 85), " 15 percent merging speed: %.2f" % np.percentile(merging_speed, 15))
x = range(10800)
np.seterr(divide='ignore',invalid='ignore')
plt.style.use('dark_background')

ax1 = fig.add_subplot(131)
ax1.grid(True)
plt.plot(x, merging_speed)
plt.title("Merging speed")
plt.sca(ax1)

ax2 = fig.add_subplot(132)
ax2.grid(True)
plt.plot(x, twt_list)
plt.title("Total travel time")
plt.sca(ax2)

plt.show()
traci.close(False)
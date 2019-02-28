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

def _getmergingspeed():
    global lanearea_ob
    ms = list()
    for lane in lanearea_ob:
        ms.append(traci.lanearea.getLastStepMeanSpeed(lane))
    meanspeed = np.mean(ms)
    return meanspeed
    
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

def _gettotaltraveltime():
    global lane_list
    traveltime = list()
    for lane in lane_list:
        if "merging" in lane:
            traveltime.append(traci.lane.getTraveltime(lane))
    ans = np.sum((traveltime))
    #print("saturation:" + str(ans))
    return ans

for _ in range(10000):
    traci.simulationStep()
    ms = _getmergingspeed()
    sat = _getsaturation()
    ttt = _gettotaltraveltime()
    print("Meanspeed: %.2f" % ms, " Saturation: %.4f" % sat, " Total travel time: %.4f" % ttt)
    traci.close(False)
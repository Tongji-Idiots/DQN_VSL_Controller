# DQN_VSL_Controller
This is a research project aiming at introducing Deep-Q-Learning methonds to help improve variable speed limitation management in traffic control system.

Log Dec 17th, 2018:
Objects to finish including filling up unfinished structure and building up connections between traci and the structure.

Log Jan 7th, 2019:
The framework has been constructed. However, the evaluation system needs to be fill in and the connection shall be confirmed working.

Log Jan 28th, 2019:
This structure has referenced to gym-based environment. Thus, the construction and wrapping methods must follow an instruction foward by the credits of gym.

Log Feb 9th, 2019:
1. Reward function modified.
previous:
    def step_reward(self):

        threshold = 101                   #it needs to be modified
        lane_speed=[0]*len(self.lane_list)
        i = 0

        for lane in self.lane_list:
            cur_speed_sum = 0
            for vehicle in self.vehicle_list[self.run_step][lane]:
                cur_speed_sum += traci.vehicle.getSpeed(vehicle)

            lane_speed[i]=cur_speed_sum / len(self.vehicle_list[self.run_step][lane])
            i += 1

        queue_len = [0] * len(self.lane_list)
        i = 0
        for lane in self.lane_list:
            j = len(self.vehicle_position[self.run_step][lane])
            while True:
                if self.vehicle_position[self.run_step][lane][j]==1:
                    queue_len[i]+=1
                else:
                    break
            i+=1

        i=0
        vehicle_sum = 0
        queue_len_sum = 0
        while i < len(self.lane_list):
            queue_len_sum+=queue_len[i]
            i+=1
        
        for lane in self.lane_list:
            vehicle_sum += len(traci.lane.getLastStepVehicleIDs(lane))

        U = queue_len_sum + vehicle_sum

        min_speed = min(lane_speed)

        if min_speed > threshold:
            return 0
        else:
            return -1*U/3600

current:
    def step_reward(self):

        queue_len = [0] * len(self.lane_list)
        i = 0
        for lane in self.lane_list:
            j = len(self.vehicle_position[self.run_step][lane])
            while True:
                if self.vehicle_position[self.run_step][lane][j]==1:
                    queue_len[i]+=1
                else:
                    break
            i+=1

        i=0
        vehicle_sum = 0
        queue_len_sum = 0
        while i < len(self.lane_list):
            queue_len_sum+=queue_len[i]
            i+=1
        
        for lane in self.lane_list:
            vehicle_sum += len(traci.lane.getLastStepVehicleIDs(lane))

        U = queue_len_sum + vehicle_sum
        
        return -(1 * U/3600 - self.pre_reward)
2. Sumo connection modification needs to be confirmed
original:
        if evaluation == False:
            
            N_SIM_TRAINING = 20
            random_seeds_training = np.random.randint(low=0, high=1e5, size= N_SIM_TRAINING)
            traci.start([self.sumoBinary, '-c', self.projectFile + 'ramp.sumo.cfg', '--start','--seed', str(np.random.choice(random_seeds_training)), '--quit-on-end'], label='training')
            self.scenario = traci.getConnection('training')
        else:
            
            N_SIM_EVAL = 3
            random_seeds_eval = np.random.randint(low=0, high=1e5, size= N_SIM_EVAL)
            traci.start([self.sumoBinary, '-c', self.projectFile + 'ramp.sumo.cfg', '--start','--seed', str(np.random.choice(random_seeds_eval)), '--quit-on-end'], label='evaluation')
            self.scenario = traci.getConnection('evaluation')

modified:(correct?)
    def reset(self, evaluation = self.evaluation):
        # Reset simulation with the random seed randomly selected the pool.
        if evaluation == False:
            seed = self.seed()[1]
            traci.start([self.sumoBinary, '-c', self.projectFile + 'ramp.sumo.cfg', '--start','--seed', str(seed), '--quit-on-end'], label='training')
            self.scenario = traci.getConnection('training')
        else:
            seed = self.seed()[1]
            traci.start([self.sumoBinary, '-c', self.projectFile + 'ramp.sumo.cfg', '--start','--seed', str(seed), '--quit-on-end'], label='evaluation')
            self.scenario = traci.getConnection('evaluation')

        self.warm_up_simulation()

        self.run_step = 0

        return self.update_observation()

3. Model saving function needs to be appended.

Log Feb 11th, 2019
1. step_reward modified.
previous:
def step_reward(self):

        queue_len = [0] * len(self.lane_list)
        i = 0
        for lane in self.lane_list:
            j = len(self.vehicle_position[self.run_step][lane])
            while True:
                if self.vehicle_position[self.run_step][lane][j] == 1:
                    queue_len[i] += 1
                else:
                    break
            i+=1

        i = 0
        vehicle_sum = 0
        queue_len_sum = 0
        while i < len(self.lane_list):
            queue_len_sum+=queue_len[i]
            i += 1
        
        for lane in self.lane_list:
            vehicle_sum += len(traci.lane.getLastStepVehicleIDs(lane))

        U = queue_len_sum + vehicle_sum
        
        return -(1 * U/3600 - self.pre_reward)

modified:
def step_reward(self):
        #Using waiting_time to present reward.

        wt = list()
        for lane in self.lane_list:
            #print(traci.lane.getWaitingTime(lane))
            wt.append(traci.lane.getWaitingTime(lane))
        self.waiting_time += np.sum(wt)
        reward = -10 if np.sum(wt) != 0 else 1
        #print(reward)
        return reward

Finally, it's all done!

Feb.22, 2019:
1. reward function changed:
    def step_reward(self):
        reward = 0.0
        episode = self.is_episode()
        if episode:
            reward += -1
        else:
            reward += 0.1
        wt = list()
        for lane in self.lane_list:
            wt.append(traci.lanearea.getLastStepMeanSpeed(lane))
        meanspeed = np.mean(wt)
        if meanspeed - self.meanspeed >= 2.77:
            reward += 1
        return reward, episode

    But failed

2. reward function changed:
    def _getmeanspeed(self):
        ms = list()
        for lane in self.lanearea_dec_list:
            ms.append(traci.lanearea.getLastStepMeanSpeed(lane))
        meanspeed = np.mean(ms)
        return meanspeed

    def step_reward(self):
        #Using waiting_time to present reward.
        reward = 0.0
        self.is_done = self.is_episode()
        if self.is_done:
            reward -= 1
        else:
            ms = self._getmeanspeed()
            if ms - self.meanspeed >= 0:
                reward += 0.1
            reward += 0.01
            self.meanspeed = ms
        return reward

3. reward function changed:
    def _getcongestionratio(self):
        for lanearea_dec in self.lanearea_dec_list:
            dec_length = 0.0
            jam_length = 0.0
            dec_length += traci.lanearea.getLength(lanearea_dec)
            jam_length += traci.lanearea.getJamLengthMeters(lanearea_dec)
        ratio = jam_length / dec_length
        return ratio
    
    def _getmeanspeed(self):
        ms = list()
        for lane in self.lane_list:
            ms.append(traci.lane.getLastStepMeanSpeed(lane))
        meanspeed = np.mean(ms)
        return meanspeed
    
    def _transform(self, x):
        return 1/(1 + np.exp(-x))

    def step_reward(self):
        #Using waiting_time to present reward.
        reward = 0.5
        speedfactor = self._getmeanspeed() - self.meanspeed
        wtfactor = self._getwaitingtime() - self.waiting_time
        ratiofactor = self._getcongestionratio() - self.ratio
        reward += self._transform(speedfactor)  - self._transform(wtfactor) - ratiofactor
        return reward
4. reward
    def _getwaitingtime(self):
        wt = list()
        for lane in self.lane_list:
            #print(traci.lane.getWaitingTime(lane))
            wt.append(traci.lane.getWaitingTime(lane))
        waiting_time = np.sum(wt)
        return waiting_time
    
    def _getcongestionratio(self):
        for lanearea in self.lanearea_ob:
            dec_length = 0.0
            jam_length = 0.0
            dec_length += traci.lanearea.getLength(lanearea)
            jam_length += traci.lanearea.getJamLengthMeters(lanearea)
        ratio = jam_length / dec_length
        return ratio
    
    def _getmeanspeed(self):
        ms = list()
        for lane in self.lane_list:
            ms.append(traci.lane.getLastStepMeanSpeed(lane))
        meanspeed = np.mean(ms)
        return meanspeed

    def step_reward(self):
        #Using waiting_time to present reward.
        reward = 0.0
        wtfactor = (self._getwaitingtime() - self.waiting_time)/self.waiting_time
        ratiofactor = (self._getcongestionratio() - self.ratio)/self.ratio
        reward += 1 - 2 * (wtfactor + ratiofactor)
        return reward

5. reward
    def _delaytime(self):
        ms = list()
        for lane in self.lane_list:
            ms.append(traci.lane.getLastStepMeanSpeed(lane))
        meanspeed = np.mean(ms)
        targetspeed = 22.22
        delaytime = np.sum(self.lane_length) * (1 / meanspeed - 1/targetspeed)
        return delaytime
    
    def _getsaturation(self):
        saturation = list()
        for lane in self.lane_list:
            saturation.append(traci.lane.getLastStepVehicleNumber(lane)/(traci.lane.getLength(lane) / 5))
        ans = np.mean(saturation)
        print("saturation:" + str(ans))
        return ans

    def _transformedtanh(self, x):
        return (np.exp(x/2) - np.exp(x/2))/(np.exp(x/2) + np.exp(x/2))
    
    def step_reward(self):
        #Using waiting_time to present reward.
        reward = 0.0
        reward += -(self._delaytime() - self.delaytime)
        self.delaytime = self._delaytime()
        return reward

6.reward 
def _getmergingspeed(self):
        ms = list()
        for lane in self.lanearea_ob:
            ms.append(traci.lanearea.getLastStepMeanSpeed(lane))
        meanspeed = np.mean(ms)
        return meanspeed
    
    def _getsaturation(self):
        saturation = list()
        for lane in self.lane_list:
            saturation.append(traci.lane.getLastStepVehicleNumber(lane)/(traci.lane.getLength(lane) / 5))
        ans = np.mean(saturation)
        #print("saturation:" + str(ans))
        self.saturation = ans

    def _transformedtanh(self, x, alpha):
        return (np.exp(-x/alpha) - np.exp(x/alpha))/(np.exp(x/alpha) + np.exp(-x/alpha))
    
    def _transformedsigmoid(self, x, alpha):
        return 1/(1 + np.exp(x/alpha))
    
    def step_reward(self):
        #Using waiting_time to present reward.
        self._getsaturation()
        delta_speed = self._getmergingspeed() - 22.22
        speed_reward = 1.5 - self._transformedsigmoid(delta_speed, 1)
        satur_discount = 2 * (0.75**2 - self.saturation ** 2)
        self.mergingspeed = self._getmergingspeed()
        if self.saturation >= 0.75 or self.mergingspeed <= 0.1:
            reward = -1.0
        else:
            reward = 0.1 * speed_reward * np.clip(satur_discount, -1, 1) * 0.06
        return reward

7. simplify observation space
original:
    def update_observation(self):
        state = np.empty((1, 3* len(self.lane_list), 441), dtype = np.float32)
        #self.state_shape = torch.from_numpy(state).shape if device == "cuda" else state.shape

        vehicle_position = np.zeros((len(self.lane_list),441),dtype = np.float32)
        vehicle_speed = np.zeros((len(self.lane_list),441),dtype = np.float32)
        vehicle_acceleration = np.zeros((len(self.lane_list),441),dtype = np.float32)

        for lane in self.lane_list:
            lane_index = self.lane_list.index(lane)
            lane_len = traci.lane.getLength(lane)
            lane_stop = int (lane_len / VEHICLE_MEAN_LENGTH)
            for i in range(lane_stop, 440):
                vehicle_position[lane_index][i] = vehicle_speed[lane_index][i] = vehicle_acceleration[lane_index][i] = -1.0

        current_step_vehicle = list()
        for lane in self.lane_list:
            current_step_vehicle += (self.vehicle_list[self.run_step][lane])

        for vehicle in current_step_vehicle:
            vehicle_in_lane = traci.vehicle.getLaneID(vehicle)
            lane_index = self.lane_list.index(vehicle_in_lane)
            vehicle_pos= traci.vehicle.getPosition(vehicle)
            lane_shape = traci.lane.getShape(vehicle_in_lane)
            vehicle_index = abs(int((vehicle_pos[0]-lane_shape[0][0])/VEHICLE_MEAN_LENGTH))

            vehicle_position[lane_index][vehicle_index] = 1.0
            vehicle_speed[lane_index][vehicle_index] = traci.vehicle.getSpeed(vehicle) 
            vehicle_acceleration[lane_index][vehicle_index] = traci.vehicle.getAcceleration(vehicle)
        
        state[0] = np.concatenate((vehicle_position, vehicle_speed, vehicle_acceleration), axis= 0)
        del current_step_vehicle
        return state
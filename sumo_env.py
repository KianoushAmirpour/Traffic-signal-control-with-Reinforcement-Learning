import traci
import numpy as np

# phases according to environment\intersection.net.xml
NS_GREEN = 0  # action 0
NS_YELLOW = 1
NSL_GREEN = 2  # action 1
NSL_YELLOW = 3
EW_GREEN = 4  # action 2
EW_YELLOW = 5
EWL_GREEN = 6  # action 3
EWL_YELLOW = 7

class Environment:
    def __init__(self, traffic_generator, sumo_cmd, max_step_per_episode,
                 num_cells_for_edge, num_actions, yellow_duration, green_duration
                 ):
        
        self.traffic_generator = traffic_generator
        self.sumo_cmd = sumo_cmd
        
        self.max_step = max_step_per_episode
        self.num_cells = num_cells_for_edge
        self.num_actions = num_actions
        self.yellow_duration = yellow_duration
        self.green_duration = green_duration
        
    def start(self, episode):
        # for each episode create a new environment\routes.rou.xml
        self.traffic_generator.generate_random_traffic(episode)
        
        # start the simulation
        traci.start(self.sumo_cmd)
         
    def get_state_observation(self):
        """
        This class will return state observations for each traffic light based on 
        the traffic which was generated for this eoisode in the form of a matrix
        """
        # variables to store TL data
        num_cars = np.zeros((self.num_cells * 2 + 4, self.num_cells * 2 + 4))
        avg_speed = np.zeros((self.num_cells * 2 + 4, self.num_cells * 2 + 4))
        cumulated_waiting_time = np.zeros((self.num_cells * 2 + 4, self.num_cells * 2 + 4))
        num_queued_cars = np.zeros((self.num_cells * 2 + 4, self.num_cells * 2 + 4))
        
        # helper dict to map lane group to specific rows and columns of the matrix that represents the state
        lanegroup_to_rc ={1:11, 2:12, 3:13, 4:14, 5:11, 6:12, 7:13, 8:14}
    
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            car_speed = traci.vehicle.getSpeed(car_id)
            car_waiting_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            car_lane_pose = traci.vehicle.getLanePosition(car_id)
            car_lane_id = traci.vehicle.getLaneID(car_id)
            
            # reversing the position of vehicles, 0 will be the closest 
            car_lane_pose = 500 - car_lane_pose
            
            # dividing the length of each edge to cells, according to num_cells
            if car_lane_pose < 7:
                lane_cell = 0
            elif car_lane_pose < 14:
                lane_cell = 1
            elif car_lane_pose < 21:
                lane_cell = 2
            elif car_lane_pose < 28:
                lane_cell = 3
            elif car_lane_pose < 35:
                lane_cell = 4
            elif car_lane_pose < 50:
                lane_cell = 5
            elif car_lane_pose < 100:
                lane_cell = 6
            elif car_lane_pose < 300:
                lane_cell = 7
            elif car_lane_pose < 700:
                lane_cell = 8
            elif car_lane_pose <= 1000:
                lane_cell = 9
            
            # get the lane group and whether this incoming road will be a row or column for TL_1
            if car_lane_id in ["E-TL_0", "E-TL_1"]:
                lane_group, rc = 1, "row"
            elif car_lane_id == "E-TL_2":
                lane_group, rc = 2, "row"
            elif car_lane_id == "W-TL_2":
                lane_group, rc = 3, "row"
            elif car_lane_id in ["W-TL_0", "W-TL_1"]:
                lane_group, rc = 4, "row"
            
            elif car_lane_id in ["N-TL_0", "N-TL_1"]:
                lane_group, rc = 5, "col"
            elif car_lane_id == "N-TL_2":
                lane_group, rc = 6, "col"
            elif car_lane_id == "S-TL_2":
                lane_group, rc = 7, "col"
            elif car_lane_id in ["S-TL_0", "S-TL_1"]:
                lane_group, rc = 8, "col"
           
            else:
                lane_group = -1
            
            if lane_group in [3, 4, 5, 6]:
                temp_cell = 9 - lane_cell
                if rc == "row":
                    index = (lanegroup_to_rc[lane_group], temp_cell)
                elif rc == "col":
                    index = (temp_cell, lanegroup_to_rc[lane_group])
            
                valid_Car = True        
                
            elif lane_group in [1, 2, 7, 8]:
                temp_cell = 14 + lane_cell
                if rc == "row":
                    index = (lanegroup_to_rc[lane_group], temp_cell)
                elif rc == "col":
                    index = (temp_cell, lanegroup_to_rc[lane_group])
                
                valid_Car = True
        
            else:
                valid_Car = False
            
            if valid_Car:
                num_cars[index[0], index[1]] += 1
                avg_speed[index[0], index[1]] += car_speed
                if car_speed < 0.1:
                    num_queued_cars[index[0], index[1]] += 1
                cumulated_waiting_time[index[0], index[1]] += car_waiting_time

        for i in range(0, (self.num_cells * 2 + 4)):
            for j in range(0, (self.num_cells * 2 + 4)):
                if num_cars[i , j] > 1:
                    avg_speed[i, j] /= num_cars[i , j]

        TL_state = np.concatenate((num_cars, avg_speed, num_queued_cars, cumulated_waiting_time)).reshape(4, self.num_cells * 2 + 4, self.num_cells * 2 + 4)
        
        return TL_state       

    def get_waiting_time(self, waiting_times):
        
        incomming_roads = ["W-TL", "N-TL", "E-TL", "S-TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            raod_id = traci.vehicle.getRoadID(car_id)
            if raod_id in incomming_roads:
                waiting_times[car_id] = wait_time
            else:
                if car_id in waiting_times:
                    del waiting_times[car_id]
        total_waiting_time = sum(waiting_times.values())
        return total_waiting_time        
        
    def get_queue_length(self):
        
        incomming_roads = ["W-TL", "N-TL", "E-TL", "S-TL"]       
        queue_length = 0
        for road in incomming_roads:
            queue_length += traci.edge.getLastStepHaltingNumber(road)
        return queue_length
    
    def set_yellow_phase(self, old_action):
        
        yellow_phase_code = old_action * 2 + 1
        traci.trafficlight.setPhase("TL", yellow_phase_code)         
        
    def set_green_phase(self, action):
        
        if action == 0:
            traci.trafficlight.setPhase("TL", NS_GREEN)
        elif action == 1:
            traci.trafficlight.setPhase("TL", NSL_GREEN)
        elif action == 2:
            traci.trafficlight.setPhase("TL", EW_GREEN)
        elif action == 3:
            traci.trafficlight.setPhase("TL", EWL_GREEN)     
        
    def simulate(self, step, max_step, steps_todo):
        if step + steps_todo > max_step:
            steps_todo = max_step - step
        while steps_todo > 0:
            traci.simulationStep()
            step += 1
            steps_todo -= 1
            queue_length = self.get_queue_length()
            # self.sum_queue_length += queue_length
            # sum_waiting_time += queue_length
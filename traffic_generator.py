import math
import random
import numpy as np


class RandomTraffic:
    
    def __init__(self, num_cars_to_generate, max_step):
        
        self._num_cars_to_generate = num_cars_to_generate
        self._max_step = max_step

    def generate_random_traffic(self, seed):
        
        """generate random traffic based on weibull distribution and assgin them to routes
        """
        random.seed(seed)
        # ran_gen_value = np.random.weibull(2, self._num_cars_to_generate)
        ran_gen_value = np.random.uniform(0,1, self._num_cars_to_generate)
        ran_gen_value = np.sort(ran_gen_value)
        
        min_ran_gen_value = math.floor(ran_gen_value[1])
        max_ran_gen_value = math.ceil(ran_gen_value[-1])
        
        arrival_times = []
        min_step = 0
        max_step = self._max_step  
        
        for value in ran_gen_value:
                arrival_times = np.append(arrival_times, ((self._max_step - min_step) / (max_ran_gen_value - min_ran_gen_value)) * (value - max_ran_gen_value) + self._max_step)

        arrival_times = np.rint(arrival_times)  
        
        with open("intersection/routes.rou.xml", "w") as routes:
            print("""<routes>
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />
            <route id="N-W" edges="N-TL TL-W"/> 
            <route id="N-S" edges="N-TL TL-S"/>
            <route id="N-E" edges="N-TL TL-E"/> 
        
            <route id="S-E" edges="S-TL TL-E"/>
            <route id="S-N" edges="S-TL TL-N"/>
            <route id="S-W" edges="S-TL TL-W"/> 
            
            <route id="W-S" edges="W-TL TL-S"/>
            <route id="W-E" edges="W-TL TL-E"/>
            <route id="W-N" edges="W-TL TL-N"/> 
            
            <route id="E-N" edges="E-TL TL-N"/>
            <route id="E-W" edges="E-TL TL-W"/>
            <route id="E-S" edges="E-TL TL-S"/> """, file=routes)

            for car_counter, step in enumerate(arrival_times):
                straight_right_or_left = np.random.uniform()
                if straight_right_or_left < 0.70:  
                    route_choice = np.random.randint(1, 9)  # choose a random source & destination
                    if route_choice == 1:
                        print('    <vehicle id="N-W_%i" type="standard_car" route="N-W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    
                    elif route_choice == 2:
                        print('    <vehicle id="N-S_%i" type="standard_car" route="N-S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    
                    elif route_choice == 3:
                        print('    <vehicle id="S-E_%i" type="standard_car" route="S-E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    
                    elif route_choice == 4:
                        print('    <vehicle id="S-N_%i" type="standard_car" route="S-N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    
                    elif route_choice == 5:
                        print('    <vehicle id="W-S_%i" type="standard_car" route="W-S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    
                    elif route_choice == 6:
                        print('    <vehicle id="W-E_%i" type="standard_car" route="W-E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    
                    elif route_choice == 7:
                        print('    <vehicle id="E-N_%i" type="standard_car" route="E-N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    
                    elif route_choice == 8:
                        print('    <vehicle id="E-W_%i" type="standard_car" route="E-W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    
                        
                else:
                    route_choice = np.random.randint(1, 5)
                    if route_choice == 1:
                        print('    <vehicle id="N-E_%i" type="standard_car" route="N-E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    
                    elif route_choice == 2:
                        print('    <vehicle id="S-W_%i" type="standard_car" route="S-W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
            
                    elif route_choice == 3:
                        print('    <vehicle id="W-N_%i" type="standard_car" route="W-N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
            
                    elif route_choice == 4:
                        print('    <vehicle id="E-S_%i" type="standard_car" route="E-S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)     
            
            print("</routes>", file=routes)
            
            
if __name__ == "__main__":
    
    random_traffic = RandomTraffic(2000, 5000)
    random_traffic.generate_random_traffic(100)
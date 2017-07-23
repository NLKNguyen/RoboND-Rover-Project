import numpy as np
import random
from scipy.spatial import distance


def is_initialize_state(Rover):
    return Rover.mode in ['start']

def initialize_state(Rover):
    if Rover.mode == 'start':
        if Rover.starting_pos is None:
            Rover.starting_pos = Rover.pos
        
        if is_near_front_wall(Rover) or is_stuck(Rover):
            Rover.mode = 'travel'
        else:
            maintain_high_speed(Rover)

    return Rover

def save_starting_position(Rover):
    Rover.starting_pos = Rover.pos

def is_travel_state(Rover):
    return Rover.mode in ['travel', 'break_loop']

def is_near_front_wall(Rover):
    return Rover.front_wall_distance < 25

def brake_until_stop(Rover):
    Rover.throttle = 0
    Rover.brake = Rover.brake_set
    Rover.steer = 0
    if Rover.vel == 0:
        return True
    else:
        return False

def is_sample_in_sight(Rover):
    return Rover.rock_size > 0


def maintain_moderate_speed(Rover):
    Rover.brake = 0    
    Rover.throttle = Rover.throttle_set / 2
    if Rover.vel > Rover.max_vel / 2:
        Rover.throttle = 0

def maintain_high_speed(Rover):
    Rover.brake = 0
    Rover.throttle = Rover.throttle_set
    if Rover.vel > Rover.max_vel:
        Rover.throttle = 0

def steer_left(Rover):
    Rover.steer = np.clip(Rover.steer + 15, -15, 15)

def steer_left_lightly(Rover):
    Rover.steer = 10


def is_near_left_wall(Rover):
    return Rover.left_wall_distance < 20

def is_too_close_to_left_wall(Rover):
    return Rover.left_wall_distance < 10

def is_too_close_to_front_wall(Rover):
    return Rover.front_wall_distance < 10

def is_near_right_wall(Rover):
    return Rover.right_wall_distance < 20

def is_too_close_to_right_wall(Rover):
    return Rover.right_wall_distance < 10


def is_too_far_from_left_wall(Rover):
    return Rover.left_wall_distance > 35

def crawl_to_left_wall(Rover):
    if is_near_left_wall(Rover):
        steer_right(Rover)
    elif is_too_far_from_left_wall(Rover):               
        steer_left(Rover)
    else:
        Rover.steer = 5
    return Rover


def is_sample_nearby(Rover):
    return Rover.rock_size > 0 and Rover.rock_angle > -15 and Rover.rock_dist < 40

def finish_collecting(Rover):
    return Rover.samples_found >= 6

def is_finalize_state(Rover):
    return Rover.mode in ['return_home', 'stop']

def steer_toward_starting_point(Rover):
    x1 = Rover.pos[0]
    y1 = Rover.pos[1]
    x2 = Rover.starting_pos[0]
    y2 = Rover.starting_pos[1]
    yaw_angle_to_target = (np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi - Rover.yaw) % 360  #- 180
    if abs(yaw_angle_to_target) < 5:
        Rover.steer = 0
        return True
    else:
        Rover.steer = np.clip(yaw_angle_to_target, -15, 15)
        return False    

def is_near_starting_position(Rover):
    return distance.euclidean(Rover.pos, Rover.starting_pos) < 5

def finalize_state(Rover):    
    if Rover.mode == 'return_home':
        if is_stuck(Rover):
            Rover.mode = 'unstuck_on_return'
        elif is_near_starting_position(Rover):
            Rover.mode = 'stop'
        else:
            if is_near_front_wall(Rover):
                maintain_moderate_speed(Rover)
            else:
                maintain_high_speed(Rover)

            steer_toward_starting_point(Rover)

            if is_near_left_wall(Rover):
                steer_right(Rover)
            elif is_near_right_wall(Rover):
                steer_left(Rover)

            if is_too_close_to_front_wall(Rover):
                if brake_until_stop(Rover):
                    Rover.mode = 'turn_away_on_return'

    elif Rover.mode == 'stop':
        if brake_until_stop(Rover):
            stand_still(Rover)
            Rover.mode = 'idle'

    return Rover



def travel_state(Rover):
    if Rover.mode == 'travel':
        if is_stuck(Rover):
            Rover.mode = 'unstuck_on_travel'
        elif is_circling(Rover):
            Rover.mode = 'break_loop'
        elif finish_collecting(Rover):
            Rover.mode = 'return_home'
        else:        
            if is_near_front_wall(Rover):
                maintain_moderate_speed(Rover)
                if not is_too_close_to_right_wall(Rover):
                    steer_right(Rover)
                else:
                    if brake_until_stop(Rover):
                        Rover.mode = 'turn_away_on_travel' # out of "travel" state and into "avoid" state
            else:
                if is_sample_in_sight(Rover):
                    maintain_moderate_speed(Rover)
                    if is_sample_nearby(Rover):
                        if brake_until_stop(Rover):
                            Rover.target_rock_pos = Rover.rock_pos
                            Rover.mode = 'approach_sample'
                    else:
                        crawl_to_left_wall(Rover)
                else:
                    maintain_high_speed(Rover)
                    crawl_to_left_wall(Rover)
    elif Rover.mode == 'break_loop':
        Rover.steer = 0
        if is_stuck(Rover):
            Rover.mode = 'travel'
        else:
            if is_near_front_wall(Rover):
                Rover.mode = 'travel'
            else:
                maintain_high_speed(Rover)
            

    return Rover


def is_avoid_state(Rover):
    return Rover.mode in ['turn_away_on_travel', 'turn_away_on_return', 'unstuck_on_travel', 'unstuck_on_return', 'unstuck_on_pickup']

def stand_still(Rover):
    Rover.throttle = 0
    Rover.steer = 0
    Rover.brake = 0


def spin_back(Rover):
    Rover.brake = 0
    
    if Rover.spin_back_counter == 0:
        Rover.throttle = random.choice([0, 0, 0, 0, -0.1, -0.2, -0.3, -0.4]) # more chance for it to turn around without moving
        Rover.steer = random.choice([-15, -10, -5, 5, 10, 15])
        Rover.spin_back_counter = random.choice([50, 100, 200, 300, 500])
    
    if Rover.spin_back_counter > 0:
        Rover.spin_back_counter -= 1
        # if is_stuck(Rover): # if stuck while reversing, then go forward instead
        #     Rover.throttle *= -1
        
    if Rover.spin_back_counter == 0:
        Rover.throttle = 0
        Rover.steer = 0
        return True
    else:
        return False    

def is_stuck(Rover, dist = None):
    if Rover.marked_pos is None:
        Rover.marked_pos = Rover.pos
        Rover.unmoveable_counter = 200

    if Rover.unmoveable_counter > 0:
         Rover.unmoveable_counter -= 1
         return False # not enough time to know if it's stuck yet, so assume not
    else: 
        # waited long enough for unmoveable_counter to get down to zero,
        # now calculate if it moved anywhere ever since.        
        delta_x = Rover.pos[0] - Rover.marked_pos[0]
        delta_y = Rover.pos[1] - Rover.marked_pos[1]

        Rover.marked_pos = None
        if dist is None:
            dist = 0.05
        return abs(delta_x) < dist and abs(delta_y) < dist # == true if it didn't move much


    # if abs(Rover.vel) < 0.1:
    #     if Rover.throttle != 0:
    #         Rover.unmoveable_counter += 1

    # if Rover.unmoveable_counter > 100:
    #     Rover.unmoveable_counter = 0
    #     return True
    # else:
    #     if Rover.throttle == 0:
    #         Rover.unmoveable_counter = 0 
    #     return False

def is_circling(Rover):
    if Rover.steer == 0:
        return False

    if  Rover.previous_steer is not None and Rover.previous_steer == Rover.steer:
        Rover.continuous_steer_counter += 1
    else:
        Rover.previous_steer = Rover.steer
        Rover.continuous_steer_counter = 0


    if Rover.continuous_steer_counter > 500:
        Rover.continuous_steer_counter = 0
        return True
    else:
        if Rover.steer != Rover.previous_steer:
            Rover.continuous_steer_counter = 0
            Rover.previous_steer = None 
        return False




def turn_away_until_clear(Rover):
    if brake_until_stop(Rover):
        if steer_until_clear(Rover):
            stand_still(Rover)
            return True
    return False


def brake_until_stop(Rover):
    Rover.brake = Rover.brake_set
    Rover.steer = 0    
    Rover.throttle = 0    
    if Rover.vel == 0:
        Rover.brake = 0
        return True
    else:
        return False

def steer_until_clear(Rover):
    Rover.brake = 0
    Rover.steer = -30        
    if Rover.front_wall_distance > 30:
        Rover.steer = 0        
        Rover.throttle = 0
        return True
    else:
        return False

def avoid_state(Rover):    
    if Rover.mode == 'turn_away_on_travel':
        if turn_away_until_clear(Rover):
            Rover.mode = 'travel'
            
    elif Rover.mode == 'turn_away_on_return':
        if turn_away_until_clear(Rover):
            Rover.mode = 'return_home'

    elif Rover.mode == 'unstuck_on_travel':
        if is_circling(Rover):
            Rover.mode = 'break_loop'               
        elif spin_back(Rover):        
            Rover.mode = 'travel'

    elif Rover.mode == 'unstuck_on_return':                
        if is_circling(Rover):
            Rover.mode = 'break_loop'               
        elif spin_back(Rover):        
            Rover.mode = 'return_home'

    elif Rover.mode == 'unstuck_on_pickup':                
        if is_circling(Rover):
            Rover.mode = 'break_loop'               
        elif spin_back(Rover):        
            Rover.mode = 'approach_sample'

    return Rover


def is_pickup_state(Rover):
    return Rover.mode in ['approach_sample', 'pickup_sample']


def look_for_sample(Rover):
    if is_sample_in_sight(Rover):        
        return True
    else:
        steer_left(Rover)
        return False

def maintain_slow_speed(Rover):
    Rover.brake = 0    
    Rover.throttle = Rover.throttle_set / 4
    if Rover.vel > Rover.max_vel / 4:
        Rover.throttle = 0
        Rover.brake = Rover.brake_set

def steer_toward_sample(Rover):
    if Rover.rock_size > 0:
        Rover.steer = np.clip(Rover.rock_angle, -15, 15)
        return False
    else:
        Rover.steer = 0
        return True

def steer_right(Rover):
    Rover.steer = np.clip(Rover.steer-15, -15, 15) #Steer right

def is_near_pickup_zone(Rover):
    # Rover.rock_angle > -15 means the rock is on the left side
    return Rover.rock_size > 0 and Rover.rock_angle > -15 and Rover.rock_dist < 15

def is_in_pickup_zone(Rover):
    return Rover.near_sample

def get_in_pickup_zone(Rover): 
    if is_in_pickup_zone(Rover):            
        if brake_until_stop(Rover):
            return True
    else:
        if is_stuck(Rover, 0.01): # minium delta distance is lower than default 0.05 since it's moving slower
            Rover.mode = 'unstuck_on_pickup'            
        else:
            if is_near_pickup_zone(Rover):
                maintain_slow_speed(Rover)
            else:
                maintain_moderate_speed(Rover)
            
            steer_toward_sample(Rover)
                  
    return False

def pickup(Rover):
    if Rover.picking_up:
        return False
    else: # not picking up
        if Rover.near_sample:
            Rover.send_pickup = True
            return False
        else:
            return True

def pickup_state(Rover):
    if Rover.mode == 'approach_sample':
        if not is_sample_in_sight(Rover): # somehow went pass it, then look back
            if brake_until_stop(Rover):                
                look_for_sample(Rover)
        else:
            if get_in_pickup_zone(Rover):
                Rover.mode = 'pickup_sample'
       
    elif Rover.mode == 'pickup_sample':
        if pickup(Rover):
            Rover.mode = 'travel'

    return Rover



def decision_step(Rover):
    if is_initialize_state(Rover):
        return initialize_state(Rover)

    elif is_travel_state(Rover):
        return travel_state(Rover)

    elif is_avoid_state(Rover):
        return avoid_state(Rover)

    elif is_pickup_state(Rover):
        return pickup_state(Rover)

    elif is_finalize_state(Rover):
        return finalize_state(Rover)

    return Rover

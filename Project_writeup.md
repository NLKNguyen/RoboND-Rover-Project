# Project: Search and Sample Return
--- 

[//]: # (Image References)

[image1]: ./misc/rover_image.jpg
[image2]: ./calibration_images/example_grid1.jpg
[image3]: ./calibration_images/example_rock1.jpg 
[stuck1]: ./misc/stuck1.png
[stuck2]: ./misc/stuck2.png
[stuck3]: ./misc/stuck3.png
[config]: ./misc/config.png
[process]: ./misc/process.png
[states]: ./misc/states.png



# Notebook Analysis

Notebook location: `./code/Rover_Project_Test_Notebook.ipynb`

## Color Threshold
Instead of using the provided `color_thresh` function that takes a lower bound threshold parameter, I modified to let it take both lower bound and upper bound thresholds. Only one parameter is required; the other is optional. This modified function is located in both Jupyter notebook and `./code/perception.py`

```python
def color_thresh(img, rgb_thresh_low=None, rgb_thresh_high=None):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])    
    
    thresh = None # To store binary 2D matrix
    
    if rgb_thresh_low is not None:
        if rgb_thresh_high is None:
            # above threshold
            thresh =  (img[:,:,0] >= rgb_thresh_low[0]) \
                    & (img[:,:,1] >= rgb_thresh_low[1]) \
                    & (img[:,:,2] >= rgb_thresh_low[2])
        else:
            # within thresholds
            thresh =   (img[:, :, 0] >= rgb_thresh_low[0]) & (img[:, :, 0] <= rgb_thresh_high[0]) \
                     & (img[:, :, 1] >= rgb_thresh_low[1]) & (img[:, :, 1] <= rgb_thresh_high[1]) \
                     & (img[:, :, 2] >= rgb_thresh_low[2]) & (img[:, :, 2] <= rgb_thresh_high[2])

    elif rgb_thresh_high is not None: # here rgb_thresh_low is None
            # below threshold
            thresh =  (img[:,:,0] <= rgb_thresh_high[0]) \
                    & (img[:,:,1] <= rgb_thresh_high[1]) \
                    & (img[:,:,2] <= rgb_thresh_high[2])
                        
    if thresh is not None: # thresh can be None if both thresholds are not provided
        color_select[thresh] = 1

    # Return the binary image
    return color_select
```



## process_image() function

### Perspective Transform

This is similar to the instruction.

Define four points in the "source" image (rover's vision) and map them to four points in the "destination" image (top-down map view)
```Python
dst_size = 5 
bottom_offset = 6
source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
img_width = img.shape[0]
img_height = img.shape[1]    
destination = np.float32([[img_height/2 - dst_size, img_width - bottom_offset],
    [img_height/2 + dst_size, img_width - bottom_offset],
    [img_height/2 + dst_size, img_width - 2 * dst_size - bottom_offset], 
    [img_height/2 - dst_size, img_width - 2 * dst_size - bottom_offset],
])
```

Apply perspective transform to the input image:
```python
warped = perspect_transform(img, source, destination)
```


### Object Detection

After some experiment with color values to identify objects (navigable ground, rock sample, or obstacle) in the vision, these are the color thresholds for each kind in the transformed image:

```python
navigable = color_thresh(warped, (118, 93, 89))
rock      = color_thresh(warped, (125, 102, 0), (204, 185, 78))
obstacle  = color_thresh(warped, None, (118,103,120))
```

### Convert thresholded image pixel values to rover-centric coords

Using the provided `rover_coords` function, apply for each object in the image:

```python
xpix, ypix = rover_coords(navigable)
rock_xpix, rock_ypix = rover_coords(rock)
obst_xpix, obst_ypix = rover_coords(obstacle) 
```

### Convert rover-centric pixel values to world coordinates

Using the provided `pix_to_world` function:
```python
current_index = data.count 
world_x = data.xpos[current_index]    
world_y = data.ypos[current_index]

yaw = data.yaw[current_index]
world_size = data.worldmap.shape[0]
world_scale = dst_size * 2


x_pix_world, y_pix_world = pix_to_world(xpix, ypix, world_x, world_y, yaw, world_size, world_scale)

rock_x_world, rock_y_world = pix_to_world(rock_xpix, rock_ypix, world_x, world_y, yaw, world_size, world_scale)

obstacle_x_world, obstacle_y_world = pix_to_world(obst_xpix, obst_ypix, world_x, world_y, yaw, world_size, world_scale)
```


### Update worldmap (to be displayed on right side of screen)

Increase red color (channel 0) for pixels of obstacle, green color (channel 1) for pixels of rock sample, and blue color (channel 2) for pixels of navigable ground.

This is the simplest form for the test case. In actual perception step in autonomous mode, there will be more tweaks to increase fidelity. That will be explained later.
```python
data.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
data.worldmap[rock_y_world, rock_x_world, 1] += 1
data.worldmap[y_pix_world, x_pix_world, 2] += 1
```

### Output video:

`./output/test_mapping.mp4`


# Autonomous Navigation and Mapping

### Simulation Mode

All experiments use the following configuration:
![config][config]

The FPS is typically around 25 on my machine.

### Sample Result

After running for some time in total autonomous mode:

![process][process]

Mapped over 80% with fidelity over 70% and collected 4 rocks in the above example.




### Modified files:
+ `perception.py`: modified `perception_step` & `color_thresh` and added `object_distance` function

+ `decision.py` : modified `decision_step` function.

+ `drive_rover.py` : added additional fields for algorithm purpose

+ `supporting_functions.py` : only added some logging info and do not affect the robot behavior. You can use the provided file in the Udacity's main repository.

## Algorithm overview


The robot is designed to be a left-wall crawler.

Using the vision data, it keeps track of the distances of the front wall, left wall, and right wall. Left and right walls here are just the area a bit on the left and right of the same vision image.

At the beginning, once it reaches the first wall, it tries to stay close to the left wall but not hit the wall.

If it gets too close to the front wall, then it makes a full brake and turn right until there is a clear path to continue.

If it sees a rock sample, it will approach the sample slowly and pick up the sample when it is close enough.

Once it collects enough of samples (6), then it tries to go back the starting position.

---

Those are the main idea of the robot's behavior, but there are a lot of more details added in order to get it out of difficult situations, for examples:

### **Misleading vision**
The obstacle color looks like a navigable path thus confuses the object detection mechanism: 
![Misleading vision][stuck1]

**How to address**: be able to detect if it is stuck by using a counter to keep track of unmoveable times (number of times the `decision_step` is called and the robot doesn't have positive velocity while throttle is not zero), and if the counter exceeds a threshold, then reverse and turn away unconditionally (i.e. not using vision).

The added function `is_stuck` is responsible to detect when it is stuck, and the function `spin_back` is called to get it out of this situation. The robot might gets stuck again, but since I added randomness in the way it reverse and turn away, it moves differently everytime and eventually will get free.

### **Vision is above short obstacles**
The short obstacles allow the robot's vision to see the navigable ground ahead which makes it continue to go while the obstacles block its wheels:

![Vision is above short obstacles][stuck2]


**How to address**: exactly the same as the above scenario, stuck too long => break away


### **Going in circle**

Because the robot is designed to crawl to the left wall, it tries to go to left until it meets the wall, but if it is in big area and the steering angle is fixed, it might go in a circle for a long time.

**How to address**: be able to detect if it is keeping the same steering angle for too long (some arbitrary threshold, not necessary a circle detection), then just let it go straight until it is close to a wall, then the left-crawling behavior resumes.

The function `is_circling` takes care of this detection.

## perception_step(Rover)

The first steps are similar to the Notebook test explained above:
* Perform perspective transform on visual image
* Apply color threshold to identify navigable terrain / obstacles / rock samples (also update `Rover.vision_image` for display)
* Convert map image pixel values to rover-centric coords
* Convert rover-centric pixel values to world coordinates

However, before all the above steps, I blur the image to reduce noise because I notice that doing so will have positive effect when calculating distance and angle.

### Update Rover world map

This is one of the main goals of the project. 

If the Rover is unstable (high roll or pitch), then don't update the world map because that will not map correctly. Just return the Rover object without changing anything.

```python
if not (Rover.pitch <= 2 or Rover.pitch >= 358) or not (Rover.roll <= 2 or Rover.roll >= 358):
    return Rover
```

Otherwise, update the world map. For each kind of object, I increase the intensity of its corresponding color channel and reduce the other 2 channels.

Eventually, the colors will be readjusted so that the main color for the kind of object will stand out more. 

```python
# Update color by channel depending on the kind of object
Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 5 # more red
Rover.worldmap[obstacle_y_world, obstacle_x_world, 1] -= 2 # less green
Rover.worldmap[obstacle_y_world, obstacle_x_world, 2] -= 2 # less blue

Rover.worldmap[rock_y_world, rock_x_world, 0] -= 2 # less red
Rover.worldmap[rock_y_world, rock_x_world, 1] += 5 # more green
Rover.worldmap[rock_y_world, rock_x_world, 2] -= 2 # less blue

Rover.worldmap[navigable_y_world, navigable_x_world, 0] -= 2 # less red
Rover.worldmap[navigable_y_world, navigable_x_world, 1] -= 2 # less green
Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 5 # more blue
```
I increase a higher amount for the corresponding color channel and decrease a lesser ammount for the other ones. The actual ammount is not that important, but I notice that this asymmetry yields higher fidelity than if it was symmetrical.


### Update Rover pixel distances and angles

In order to detect if Rover is close to front wall or side walls, I add a custom function `object_distance` to compute the distance to obstaces at certain degree from the Rover's view.

```python
def object_distance(dists, angles, degree):
    min_angle = (degree - 3) / 180 * np.pi
    max_angle = (degree + 3) / 180 * np.pi
    distances = dists[(angles >= min_angle) & (angles <= max_angle)]
    if len(distances) > 0:
        return np.min(distances)
    return 9999 # arbitary far away
```

Save the calculated distances to obstacle in front and sides

```python
# first obtain the distances and angles to all obstaces in view
obstacle_dists, obstacle_angles = to_polar_coords(obstacle_x_rover, obstacle_y_rover)

# save distances to obstacles at certain degree
Rover.front_wall_distance = object_distance(obstacle_dists, obstacle_angles, 0)
Rover.left_wall_distance  = object_distance(obstacle_dists, obstacle_angles, 35)
Rover.right_wall_distance = object_distance(obstacle_dists, obstacle_angles, -35)
```

For sample rock detection and pickup planning, I save the distance, angle and the position of the rock. `Rover.rock_size > 0` means there is a sample rock in view. The closer it gets to the rock, the bigger the rock size in view, but that is not important because I use `Rover.rock_dist` to measure how close it is. 
```python
# Calculate the distance and angle of the sample rock in view
rock_dists, rock_angles = to_polar_coords(rock_x_rover, rock_y_rover)

Rover.rock_size = len(rock_dists)
if Rover.rock_size > 0:
    Rover.rock_dist = np.mean(rock_dists)
    Rover.rock_angle = np.mean(rock_angles * 180 / np.pi)
    Rover.rock_pos = (np.mean(rock_x_world), np.mean(rock_y_world))
else:
    Rover.rock_dist = 0
    Rover.rock_angle = 0
    Rover.rock_pos = None
```



## decision_step(Rover)


Because the Rover's behavior can be complicated, a simple decision tree algorithm approach will soon become difficult to maintain and modify. Instead, I use a state machine approach. This is an overview of the state machine design.

There are 5 main `states`:
* **initialize**: the first state that Rover will be in when the autonomous process begins
* **travel**: the most common state where Rover navigate through the terrain
* **avoid**: find navigable path or get out of current position
* **pickup**: approach to sample rock in sight and pick it up
* **finalize**: going back to starting position when collected all rocks


```python
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
```

Each state can have multiple internal states that are called modes to easy distinguish, and `Rover.mode` stores the current mode. The purpose of having the above main states is simply to make the source code more modular. The way to change a state is to modify the `Rover.mode` to one of the modes that belong to another state.

Below is the complete state machine design where each entity is a `mode`, and the name on the arrow is the function that if succeeds (return `True`) will change the mode, otherwise it stays in the current mode.


![states][states]

Even though the nature of this system is that the `decision_step` is called many times per second, I construct all of the helper functions in such a way that makes the source code more descriptive and seem sequential, i.e. not having to concern about the continous function invokes, in order to reason about the Rover behavior more easily.

The pattern you will see a lot in `decision.py` source code is a lot of nested `if` clauses with function calls that only return True if the operation finishes successfully, otherwise the control flow will makes the exact execution path happen again next time, thus the function will get called again until the Rover is in such a configuration that makes it return True in order to move to the next operation.

For example, in the `pickup_state` function below, the function `pickup(Rover)` will make the Rover pick up the rock, which takes time and spans through hundred of `decision_step(Rover)` calls that invoke this `pickup_state(Rover)` function each time, and at the last time that the pick up operation succeeds, the result of `pickup(Rover)` will be True, so the next step can occur, which is `Rover.mode = 'travel'` to switch back to travel state.

```python
def pickup_state(Rover):
    if Rover.mode == 'approach_sample':
        if not is_sample_in_sight(Rover):
            # somehow went pass it, then look back
            if brake_until_stop(Rover):                
                look_for_sample(Rover)
        else:
            if get_in_pickup_zone(Rover):
                Rover.mode = 'pickup_sample'
       
    elif Rover.mode == 'pickup_sample':
        if pickup(Rover):
            Rover.mode = 'travel'

    return Rover
```

The function `pickup(Rover)` has similar coding pattern. In this case, `Rover.send_pickup = True` will start the pick up operation, handled by another source file, and only when the Rover is in desired configuration, it returns True.
```python
def pickup(Rover):
    if Rover.picking_up: # operation not finish yet
        return False
    else: # not picking up
        if Rover.near_sample: # if there is a rock
            Rover.send_pickup = True # trigger pick up operation
            return False # of course the rock is not picked up yet
        else: # no more rock => picked up already
            return True
```

This gives an illusion that the helper functions like `pickup(Rover)` are blocking which makes it easy to reason about the control flow, but in fact hundreds of calls occur, and they just happen to go through that the same execution path until that function returns True in order to proceed further.



# Possible Failures and Improvement

Despite some considerations for difficult scenarios explained before, there are still corner cases that the Rover might fail.

### On a slope
Since the algorithm expects a flat ground for navigation, it considers non-flat ground as Rover being unstable, thus not update the Rover. The side effect is that if it's actually on a slope, the Rover will not update the visual information for the decision step at all; therefore, Rover doesn't have its usual behavior and gets stuck. In this case, however, it knows that it is stuck, and only the `spin_back` function is called to randomly move around, hopefully to get back to the flat ground.

**How to address**: One way I think of is to have another detection mechanism to handle this situation when the ground is not flat in order to still be able to utilize the visual information but not affect the world map.

### On the way back

At this point, the Rover does not do a good job on returning to the starting position after finish collecting rocks. The current algorithm simply steers it toward to starting position without realizing any obstacles blocking its way. It stills avoid not hitting the obstacles normally, but then it steers back to the same direction. Now it is not smart enough to use the knowledge from the recorded map in order to go another direction so that, later on, the path to the destination will be more clear.

**How to address**: One obvious attempt could be to figure out a way to use the recorded map information in order to create a multi-steps planning to get from point A to B in a similar manner to a real navigation program such as Google Maps.

Another approach that I'm interested in trying is to let it go randomly (could be exactly the same as the current left-wall crawler algorithm), and when its distance to the starting point is less than a certain small threshold, which indicates that Rover is now in the center area where the Rover is dropped, then let it steer toward the starting position. There could be small obstacles on the way, but it can get around eventually. 
 

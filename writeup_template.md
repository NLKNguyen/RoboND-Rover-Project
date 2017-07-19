## Project: Search and Sample Return
### Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook). 
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands. 
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # (Image References)

[image1]: ./misc/rover_image.jpg
[image2]: ./calibration_images/example_grid1.jpg
[image3]: ./calibration_images/example_rock1.jpg 
[stuck1]: ./misc/stuck1.png
[stuck2]: ./misc/stuck2.png
[stuck3]: ./misc/stuck3.png


## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!


![alt text][image2]
### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.


#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  



![alt text][image3]




# Notebook Analysis

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

## Algorithm overview


The robot is designed as a left-wall crawler.

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

The added function `is_stuck` is responsible to detect when it is stuck, and the function `spin_away` is called to get it out of this situation. The robot might gets stuck again, but since I added randomness in the way it reverse and turn away, it moves differently everytime and eventually will get free.

### **Vision is above short obstacles**
The short obstacles allow the robot's vision to see the navigable ground ahead which makes it continue to go while the obstacles block its wheels:

![Vision is above short obstacles][stuck2]


**How to address**: exactly the same as the above scenario, stuck too long => break away


### **Going in circle**

Because the robot is designed to crawl to the left wall, it tries to go to left until it meets the wall, but if it is in big area and the steering angle is fixed, it might go in a circle for a long time.

**How to address**: be able to detect if it is keeping the same steering angle for too long (some arbitrary threshold, not necessary a circle detection), then just let it go straight until it is close to a wall, then the left-crawling behavior resumes.

The function `is_circling` takes care of this detection.

## perception_step()
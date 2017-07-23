import numpy as np
import cv2

# Identify pixels above the threshold
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

# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

# Define a function to perform a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    return warped


# Calculate the distance to object at certain degree
def object_distance(dists, angles, degree):
    min_angle = (degree - 3) / 180 * np.pi
    max_angle = (degree + 3) / 180 * np.pi
    distances = dists[(angles >= min_angle) & (angles <= max_angle)]
    if len(distances) > 0:
        return np.min(distances)
    return 9999 # arbitary far away




# Measures and calculates the fields of the rover state based on sensor data
def perception_step(Rover):
    
    # 1) Define constants like source and destination points for perspective transform
    img = Rover.img
    dst_size = 5 
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    img_width = img.shape[0]
    img_height = img.shape[1]    
    destination = np.float32([[img_height/2 - dst_size, img_width],
                      [img_height/2 + dst_size, img_width],
                      [img_height/2 + dst_size, img_width - 2 * dst_size], 
                      [img_height/2 - dst_size, img_width - 2 * dst_size],
                      ])
   
    # Blur image to reduce noise
    img = cv2.GaussianBlur(img, (11, 11), 0)

    # 2) Apply perspective transform
    warped = perspect_transform(img, source, destination)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples    
    navigable = color_thresh(warped, (118, 93, 89))
    rock      = color_thresh(warped, (125,102,0), (204,185,78))
    obstacle  = color_thresh(warped, None, (118,103,120))

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:,:,0] = obstacle * 255  # red channel
    Rover.vision_image[:,:,1] = rock * 255      # green channel
    Rover.vision_image[:,:,2] = navigable * 255 # blue channel

    # 5) Convert map image pixel values to rover-centric coords
    navigable_x_rover, navigable_y_rover = rover_coords(navigable)
    rock_x_rover, rock_y_rover           = rover_coords(rock)
    obstacle_x_rover, obstacle_y_rover   = rover_coords(obstacle)

    # 6) Convert rover-centric pixel values to world coordinates
    yaw = Rover.yaw
    world_size = Rover.worldmap.shape[0]
    world_scale = dst_size * 2
    world_x = Rover.pos[0]   
    world_y = Rover.pos[1]
    navigable_x_world, navigable_y_world = pix_to_world(navigable_x_rover, navigable_y_rover, 
                                                        world_x, world_y, yaw, world_size, world_scale)
    rock_x_world, rock_y_world           = pix_to_world(rock_x_rover, rock_y_rover, 
                                                        world_x, world_y, yaw, world_size, world_scale)
    obstacle_x_world, obstacle_y_world   = pix_to_world(obstacle_x_rover, obstacle_y_rover, 
                                                        world_x, world_y, yaw, world_size, world_scale)
    
    # 7) Update Rover worldmap (to be displayed on right side of screen)

    # if not in range of stability, then don't update the map
    if not (Rover.pitch <= 2 or Rover.pitch >= 358) or not (Rover.roll <= 2 or Rover.roll >= 358):
        return Rover
    
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

    # Update Rover pixel distances and angles

    # Calculate the distance to obstacle in front and sides
    obstacle_dists, obstacle_angles = to_polar_coords(obstacle_x_rover, obstacle_y_rover)
    
    Rover.front_wall_distance = object_distance(obstacle_dists, obstacle_angles, 0)
    Rover.left_wall_distance  = object_distance(obstacle_dists, obstacle_angles, 35)
    Rover.right_wall_distance = object_distance(obstacle_dists, obstacle_angles, -35)
    
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

    return Rover

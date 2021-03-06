import os
import io
import glob
import time
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.npyio import save 
import scipy.ndimage.filters as filters

# import custom libraries
from utils.transform import Transform
from utils.yamlparser import YamlParser

# import TensorRT wrapper class
from yolov5_head import YOLOv5HeadModel

import pycuda.autoinit

def collect_images(data_dir):
    """
    Temporary function for getting data for bird's eye view transformation
    """
    image_paths = glob.glob(data_dir +"/*.jpg")
    if not image_paths:
        raise SystemExit("There is no images in given directory!") 
    else:
        images = {}
        for image_path in image_paths:
            image_path = os.path.abspath(image_path)
            cam_id = int(image_path.split(".")[0].split("/")[-1])
            images[cam_id] = image_path

        return images

def get_scale(W, H, scale):
    """
    Computes the scales for bird-eye view image

    Parameters
    ----------
    W    : int
           Polygon ROI width
    H    : int
           Polygon ROI height
    scale: list
           [height, width]
    """

    dis_w = int(scale[1])
    dis_h = int(scale[0])
    
    return float(dis_w/W),float(dis_h/H)

def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.resize(img, dsize=(593, 1600))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

# temporary function
def create_heatmap(centroids):
    """
    Create a heatmap by using centroids with matplotlib framework.
    Then transform the matplotlib figure to numpy image

    Parameters
    ----------
    centrodis : numpy array (N x 2)
                numpy array, centroids of detected humans from aerial view

    Returns
    -------
    np_heatmap : numpy array
                 heatmap, generated from matplotlib in numpy format
    """
    
    # setup parameters for density map generation
    # disable frame, 
    fig = plt.figure(frameon=False)

    # frequency map
    frequency_map = np.full((1600, 593), 50).astype(np.int)
    # disable axes
    # fill the entire figure with content
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Increment the frequency count of frequency map
    # according to occurence of centroids with respect to frequency map
    for centroid in centroids:
        frequency_map[centroid[1], centroid[0]] += 7 # 5

        # increment the surrounding with two
        frequency_map[centroid[1] - 8:centroid[1] + 8, centroid[0] - 8: centroid[0] + 8] += 5 # INCREMENT 3, WIDTH HEIGHT

    # zero out all the pixels where there is no overlapp
    frequency_map[frequency_map == 50] = 0
    # smooth it to create a "blobby" look
    heatmap = filters.gaussian_filter(frequency_map, sigma=15)
    # generate heatmap, based on frequency map
    ax.imshow(heatmap, cmap='jet', aspect='auto')

    np_heatmap = get_img_from_fig(fig=fig, dpi=180)

    return np_heatmap

def compute_occupant_per_zone(bird_eye_view, cfg):
    """
    Segment the bird eye view into zones for occupant counting

    Parameters
    ----------
    bird_eye_view   : numpy array
                      bird-eye view of entire naplab
    cfg             : easydict object
                      dictionary, containing information of zones
    """

    ori_zone = 320
    zones_count = 5
    # zones are in this order : [zone, zone2, zone3, zone4, zone5]
    zones = [426, 320, 320, 320, 214]
    # dictionary for storing extracting occupancy count in each zone
    occup_zones = {}

    # extract occupancy count in zone 1
    zone1 = bird_eye_view[sum(zones[1:]):, :, :]
    occup_zones['1'] = len(extract_calibrated_centroids(zone1.astype(np.uint8)))
    # cv2.imwrite("zone1.jpg", zone1)

    # extract occupancy count in zone 2
    zone2 = bird_eye_view[sum(zones[2:]):sum(zones[1:]), :, :]
    occup_zones['2'] = len(extract_calibrated_centroids(zone2.astype(np.uint8)))
    # cv2.imwrite("zone2.jpg", zone2)
    
    # extract occupancy count in zone 3
    zone3 = bird_eye_view[sum(zones[3:]):sum(zones[2:]), :, :]
    occup_zones['3'] = len(extract_calibrated_centroids(zone3.astype(np.uint8)))
    # cv2.imwrite("zone3.jpg", zone3)

    # extract occupancy count in zone 4
    zone4 = bird_eye_view[sum(zones[4:]):sum(zones[3:]), :, :]
    occup_zones['4'] = len(extract_calibrated_centroids(zone4.astype(np.uint8)))
    # cv2.imwrite("zone4.jpg", zone4)

    # extract occupancy count in zone 5
    zone5 = bird_eye_view[:zones[-1], :, :]
    occup_zones['5'] = len(extract_calibrated_centroids(zone5.astype(np.uint8)))
    # cv2.imwrite("zone5.jpg", zone5)

    return occup_zones

def compute_object_centroids(boxes):
    """
    As expected from the function name, this function 
    computes estimated centroids of humans from detected head boxes

    Current approach
        _________
        |       |
        |   .   | <------ bbox midpoint is not where it appears on bird'eye view
        |___|___|
            |
            | <------ This distance is the same as height of bbox (or half of the height of the box)
            |
            . <------ Take this point instead of bbox midpoint 

    This should be suboptimal since it is hardcoded.

    Parameters
    ----------
    boxes : numpy array
            If there is no detected bboxes, shape : (0, 4)
            else, shape : (n, 4)
    
    Returns
    -------
    centroids : numpy array
                shape : (n, 2)
    """
    if len(boxes) == 0:
        return np.zeros((0, 2))
    else:
        centroids = np.zeros((len(boxes), 2))
        for i, box in enumerate(boxes):
            centroids[i][0] = float((box[0] + box[2]) *  0.5)
            # centroids[i][1] = float((box[1] + box[3]) *  0.5)
            centroids[i][1] = float(box[3] + (box[3] - box[1]))

        return centroids.astype(np.float32)

def extract_calibrated_centroids(unified_view, centroid_color=[0, 0, 255]):
    """
    Extract new centroid positions from unified_view image.

    Parameters
    ----------
    unified_view   : numpy array image in np.uint8 format
                     Image, resulted from coordinated concatenation of region bird eye view images
    centroid_color : list
                     BGR color code of the centroids of detections

    !Extra Note
    Naive implementation to get new centroids by double-looping the width and height of the image
    gives you really bad performance so finding new centroid coordinates with cv2 functions
    is optimal method.
    """

    # numpy array manipulation method
    # execution time : 0.005768s
    # centroid_x, centroid_y = np.where(np.all(unified_view==np.array(centroid_color), axis=-1))

    # cv2 thresholding method
    # execution time: 0.000844s
    upper_red = np.array(centroid_color)
    lower_red = np.array(centroid_color)
    mask  = cv2.inRange(unified_view, lower_red, upper_red)
    coord = cv2.findNonZero(mask)

    if coord is None:
        return []
    else:
        return np.squeeze(coord)

def create_unified_bird_eye_view(bird_eye_images, cfg):
    """
    Create a unified bird eye view from region bird eye view images.
    This function can only be used for NapLab unified bird eye view image generation.

    Simple Explanation
    ------------------

    The entire Naplap bird eye view size : height 1600 x  width 593
    Regions are bird-eye view of zones and in this implementation,
    we are going to concatenate to get a unified view.

    Reference `zones/NapLab Regions and Zones.jpg` for more details.

    Step1 :  Create a canvas of entire unified bird eye view
    Step2 :  Fill the images from bird_eye_images to corresponding regions one by one

    bird_eye_images must be in this order for generation of a proper unified image.
    [camera 221 view, camera 220 view, camera 218 view, camera 219 view]

    Parameters
    ----------
    bird_eye_images   : list
                        list containing numpy array images
                        [camera 221 view, camera 220 view, camera 218 view, camera 219 view]
                        [region1, region2, region3, region4]
    cfg               : Argument parser object
                        dictionary, containing information for creating

    Returns
    -------
    canvas            : numpy uint8 array
                        unified bird-eye view
    """

    # region parameters
    # region 1
    region1_height = cfg['221'].scale[0]
    region1_width  = cfg['221'].scale[1]
    # region 2
    region2_height = cfg['220'].scale[0]
    region2_width  = cfg['220'].scale[1]
    # region 3
    region3_height = cfg['218'].scale[0]
    region3_width  = cfg['218'].scale[1]
    # region 4
    region4_height = cfg['219'].scale[0]
    region4_width  = cfg['219'].scale[1] 
    

    # create white pixel canvas to be filled with 
    canvas = np.full((1600, 593 , 3), 255)
    cv_shape = canvas.shape

    # fill region 1
    # Region 1 covers two zones (extra + zone1, 60% of zone2)
    # Therefore, Region 1 is the last two zones of canvas
    canvas[cv_shape[0]- region1_height:, :region1_width, :] = bird_eye_images[0]

    # fill region 2
    # Region 2 is inbetween area of Region1 and Region4
    # Region 2 covers 40% of zone 2 and zone 3
    canvas[cv_shape[0]-(region1_height + region2_height):cv_shape[0]-region1_height, :region2_width, :] = bird_eye_images[1]

    # fill region 4
    # Region 4 covers the 80% of zone 5 and zone 4
    # Therefore, Region 4 is the first two zones of canvas
    # need to rotate the image first because camera angle is reverse from others
    rotated_image = cv2.rotate(bird_eye_images[3], cv2.ROTATE_180)
    canvas[:region4_height, :region4_width, :] =  rotated_image # be_view_images[3]

    # fill region 3
    # Region 3 is standalone part of naplab
    # it covers a small part of zone 4 and zone 5
    # need to rotate the image first because camera angle is reverse from others
    rotated_image = cv2.rotate(bird_eye_images[2], cv2.ROTATE_180)
    canvas[:region3_height, region4_width:, :] = rotated_image # be_view_images[2]

    return canvas.astype(np.uint8)

def create_region_be_view(image_paths, centroids, cfg, transform):
    """
    Generate bird eye view for regions by using centroids of detections

    Parameters
    ----------
    image_paths : str
                  Image paths
                  Images are from cameras, specifying respective region
    centroids   : numpy array
                  batch centroids of detections, made in that images
    cfg         : Argument parser object
                  dictionary, containing information for creating
                  bird eye view for regions
    transform   : instance of Transform class/object
                  Collection of transformer function
    """

    cameras = ['221', '220', '218', '219']
    bird_eye_images = []

    for idx in range(len(image_paths)):

        cur_image = cv2.imread(image_paths[idx])
        cur_image = cv2.resize(cur_image, dsize=(960, 640))

        cam = cameras[idx]
        # getting padding informations
        place = cfg[cam].place
        pad   = cfg[cam].pad
        # get polygon shaped ROI coordinates
        tl = cfg[cam].top_left
        tr = cfg[cam].top_right
        br = cfg[cam].bot_right
        bl = cfg[cam].bot_left
        # get bird eye view ccales
        be_scale = cfg[cam].scale

        # transform both base image and detected person centroids with padding function 
        padded_image, padded_centroids = transform.be_transform(cur_image, place, pad, centroids[idx])

        # test_image = np.copy(padded_image).astype(np.uint8)
        # for c in padded_centroids:
        #     cv2.circle(test_image, (int(c[0]), int(c[1])), radius=5, color=[0, 0, 255], thickness=10)
        # cv2.imwrite("padded_centroids_{}.jpg".format(idx), test_image)
            

        # change to unit8 numpy array for image viewing
        padded_image = padded_image.astype(np.uint8)

        padded_height, padded_width = padded_image.shape[:2]

        # Set scale for birds eye view
        # Bird's eye view will only show ROI
        scale_w, scale_h = get_scale(padded_width, padded_height, be_scale)

        # compute transform matrix for generating aerial bird view of centroids
        transform_matrix = transform.compute_perspective_transform(corner_points=(tl, tr, br, bl),
                                                                   width=padded_width,
                                                                   height=padded_height)
        
        # compute ground_centroids by using transform matrix
        new_centroids = transform.compute_point_perspective_transform(transform_matrix, padded_centroids)
        # finally generate bird eye view
        bird_eye_image = transform.bird_eye_view_transform(padded_image, new_centroids, scale_w, scale_h)
        # cv2.imwrite("be_view_{}.jpg".format(idx), bird_eye_image)

        bird_eye_images.append(bird_eye_image)

    return bird_eye_images

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trt", type=str, default="./checkpoints/head_yolov5_1.trt",
                        help="Generated TensorRT runtime model")
    parser.add_argument("--image_folder", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--config", type=str, default="./configs/naplab.yaml")
    parser.add_argument("--conf_threshold", type=float, default=0.3)
    parser.add_argument("--nms_threshold", type=float, default=0.5)

    return parser.parse_args()

def main(args):
    """
    main function for gluing all required functions to run the entire pipeline together

    Parameters
    ----------
    args : argument parser object
           parser object, containing all arguments to run the pipeline
    """
    trt_runtime_model = args.trt
    image_folder_path = str(args.image_folder)
    config_file = args.config
    conf_threshold = args.conf_threshold
    nms_threshold = args.nms_threshold
    save_dir = args.save_dir
     
    # check whether given files exists or not
    if not os.path.isfile(trt_runtime_model):
        raise SystemExit("ERROR: Specified TensorRT runtime model {:s} not found!".format(trt_runtime_model))
    if not os.path.exists(image_folder_path):
        raise SystemExit("ERROR: Path {:s} not found! Check the Dir!".format(image_folder_path))
    if not os.path.exists(save_dir):
        raise SystemExit("ERROR: Path {:s} not found! Check the Dir!".format(save_dir))
    if not os.path.isfile(config_file):
        raise SystemExit("ERROR: file {:s} not found! Check the file!".format(config_file))

    # De-serialize the TensorRT model
    trt_model = YOLOv5HeadModel(engine_path=trt_runtime_model, nms_thres=nms_threshold,
                                conf_thres=conf_threshold)
    # load configuration file
    cfg = YamlParser(config_file)

    # get the images for demo
    demo_images = collect_images(data_dir=image_folder_path)

    # Here is the place where we check if the dates are matched or not

    # cameras list will be provided from config file in the future
    cameras = [221, 220, 218, 219]
    
    # Perform inference with head detection model
    all_centroids = []
    images = []
    t0 = time.time()
    for cam_id in cameras:
        images.append(demo_images[cam_id])
        image_data = cv2.imread(demo_images[cam_id])
        # perform inference
        boxes = trt_model.detect(img=image_data)
        # compute estimated body centroids from detected head boxes
        centroids = compute_object_centroids(boxes=boxes[0])
        all_centroids.append(centroids)

        """
        Debug code segment: Write images and Visualize inference results

        color = [np.random.randint(0, 255), 0, np.random.randint(0, 255)]
        for i in range(len(boxes[0])):
            x1, y1, x2, y2, _, _ = boxes[0][i]
            centroid = centroids[i]
            cv2.rectangle(img=image_data, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=color, thickness=2)
            cv2.circle(image_data, ( int(centroid[0]), int(centroid[1])), 4, color, 8)

        cv2.imwrite(str(cam_id) + "_debug.jpg", image_data)
        """
    t1 = time.time()

    # create birdeye view for all images and zones
    """
    The following functions are handcrafted and needed to be tuned to work with new scenario.
    Current scenario : Naplab
    
    CAUTION:
    Images needs to be in a order mentioned by cameras.
    Later processes requires hand-craft concatenation and 
    """
    # create Transformer instance
    T = Transform()
    bird_eye_images = create_region_be_view(image_paths=images, centroids=all_centroids, cfg=cfg, transform=T)
    """
    Debug code segment: Write generated bird's eye view images
    for cam_id, be_img in zip(cameras, bird_eye_images):
        cv2.imwrite(str(cam_id) + "_be.jpg", be_img)    
    """
    t2 = time.time()
    # create a unitied bird's eye view of the entire naplab
    unified_bird_eye_image = create_unified_bird_eye_view(bird_eye_images=bird_eye_images, cfg=cfg)
    t3 = time.time()
    true_centroids = extract_calibrated_centroids(unified_view=unified_bird_eye_image)
    t4 = time.time()

    # calculate count of people at designated zones
    occup_zones = compute_occupant_per_zone(bird_eye_view=unified_bird_eye_image, cfg=cfg)
    print(occup_zones)
    t5 = time.time()

    # generate heatmap
    heatmap = create_heatmap(centroids=true_centroids)

    # write the heatmap
    cv2.imwrite(str(os.path.abspath(save_dir)) + "/heatmap.jpg", heatmap)
    t6 = time.time()

    print("Inference time : {}".format(t1 - t0))
    print("Bird's eye image generation : {}".format(t2 - t1))
    print("Unified bird's eyeview generation : {}".format(t3 - t2))
    print("Centroids calibration : {}".format(t4 - t3))
    print("Occupant count per zone : {}".format(t5 - t4))
    print("Density map generation : {}".format(t6 - t5))

    """
    Debug code segment : Write generated unified bird's eye view
    color = [np.random.randint(0, 255), 0, np.random.randint(0, 255)] 
    for centroid in true_centroids:
        cv2.circle(unified_bird_eye_image, (int(centroid[0]), int(centroid[1])), 4, color, 8)

    cv2.imwrite("entire_be_image.jpg", unified_bird_eye_image)
    """

if __name__ == "__main__":
    args = parse_args()
    t0 = time.time()
    for _ in range(20):
        main(args)
        print("-------------------------------------------------")
    t1 = time.time()

    FPS = 1/((t1 - t0)/30)
    average = (t1 -t0)/30
    print("Entire pipeline average execution time : {}".format(average))
    print("Potential FPS : {}".format(FPS))

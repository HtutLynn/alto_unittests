import cv2
import numpy as np
import math

class Transform(object):
    """
    Collection for `transform` operations which is required for the heatmap
    generation and bird eye view conversion

    1. be_transform : Pad the image for creating appropriate ROI generation
    2. compute_perspective_transform : Generates the image transformation matrix for perspective change
    3. compute_point_perspective_transform : Compute the centroids to their corresponding ground centroids
                                             for bird eye view
    4. bird_eye_view_transform : Draw the ground centroids on a canvas to look like bird eye view
    """

    @staticmethod
    def be_transform(base_image, place, pad, centroids):
        """
        Pad the image, for birdeye view image transformation.
        Also transform the centroids 

        Parameters
        ----------
        image    : numpy array
                Image, read with cv2 module
        place    : [end, start]
                padding information
                first idx  : height
                second idx : width
                `end` pad that dimension at the end.
                `start` pad that dimension at the start.
        pad      : [200, 300]
                first idx  : height
                second idx : width
        centroids: numpy array
                first idx  : width # caution! inverse with other parameters
                first idx  : height
                n x 2 array containing centroids
        """
        size = base_image.shape
        centroids = centroids
        # create a canvas for final padded image
        padded_image = np.full((size[0] + pad[0], size[1] + pad[1], size[2]), 255)

        if place[0] == 'start' and place[1] == 'start':
            padded_image[pad[0]:, pad[1]:, :] = base_image
            centroids[:, 0] += pad[1]
            centroids[:, 1] += pad[0]
        elif place[0] == 'start' and place[1] == 'end':
            padded_image[pad[0]:, :size[1], :] = base_image
            centroids[:, 1] += pad[0]
        if place[0] == 'end' and place[1] == 'start':
            padded_image[:size[0], pad[1]:, :] = base_image
            centroids[:, 0] += pad[1]
        else:
            padded_image[:size[0], :size[1], :] = base_image

        return padded_image, centroids.astype(np.int)

    @staticmethod
    def compute_perspective_transform(corner_points, width, height):
        """ Compute the transformation matrix for perspection change : Bird eye view

        Parameters
        ----------
        corner_points : numpy array
                        4 corner points selected from the image
        height, width : int 
                        size of the image

        Returns
        ------
        perspective_transform_matrix : numpy array/matrix (known as transformation matrices in Computer Graphics)
        
        """

        # Using first 4 points or coordinates for perspective transformation. The region marked by these 4 points are 
        # considered ROI. This polygon shaped ROI is then warped into a rectangle which becomes the bird eye view. 
        # This bird eye view then has the property property that points are distributed uniformally horizontally and 
        # vertically(scale for horizontal and vertical direction will be different). So for bird eye view points are 
        # equally distributed, which was not case for normal view.
        corner_points_array = np.float32(corner_points)
        # Create an array with the parameters (the dimensions) required to build the matrix
        dst = np.float32([[0, height], [width, height], [width, 0], [0, 0]])
        
        perspective_transform_matrix = cv2.getPerspectiveTransform(corner_points_array, dst)

        return perspective_transform_matrix

    @staticmethod
    def compute_point_perspective_transform(transform_matrix, centroids):
        """ Apply the perspective transformation matrix to detected centroids, to gain ground centroids for aerial view 
        
        Parameters
        ----------
        transform_matrix : numpy array
                        perspective transformation matrix
        centroids        : numpy array
                        (n x 2) array which contains the centroids points
        
        Returns
        -------
        new_centroids : list
                        list
        """
        # set up a container
        new_centroids = []
        for centroid in centroids:
            point = np.array([[[int(centroid[0]), int(centroid[1])]]], dtype="float32")
            ground_point = cv2.perspectiveTransform(point, transform_matrix)[0][0]
            new_point    = [int(ground_point[0]), int(ground_point[1])]
            new_centroids.append(new_point)

        return new_centroids

    @staticmethod
    def bird_eye_view_transform(paded_image, bot_centroids, scale_w, scale_h):
        """
        Plot bird_eye_view for detected person centroids

        Parameters
        ----------
        padded_image  : numpy array
                        Image, with appropriate white padding for better aerial view
        bot_centroids : numpy array
                        Centroids of detected persons, converted to ground centroids
        scale_w       : int
                        scaling factor for bird eye view
        scale_h      : int
                        scaling factor for bird eye view
        """

        h = paded_image.shape[0]
        w = paded_image.shape[1]

        white = (200, 200, 200) # RGB color code
        red   = (0, 0, 255)   

        # create a canvas
        # ceiling function is needed for getting proper width after scaling 
        canvas = np.zeros((math.ceil(h * scale_h), math.ceil(w * scale_w), 3), np.uint8)
        # canvas = np.zeros((int(h * scale_h), int(w * scale_w), 3), np.uint8)
        canvas[:] = white

        for idx, centroid in enumerate(bot_centroids):
            cv2.circle(canvas, ( int(centroid[0]  * scale_w), int(centroid[1] * scale_h)), 0, red, 1)

        return canvas
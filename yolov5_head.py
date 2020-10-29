from __future__ import print_function
from os import stat
import os.path as ops
from typing import Union

import cv2
import numpy as np
import tensorrt as trt
from scipy.special import expit

# import cuda functions
import pycuda.driver as cuda

def _aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
    """
    Pads input image without losing the aspect ratio of the original image

    Parameters
    ----------
    image         : numpy array
                    In BGR format
                    uint8 numpy array of shape (img_h, img_w, 3)
    width         : int
                    width of newly padded image
    height        : int
                    height of newly padded image
    interpolation : str
                    method, to be applied on the image for resizing
    
    Returns
    -------       
    canvas        : numpy array
                    float 32 numpy array of shape (height, width, 3)
    new_w         : int
                    width, of the image after resizing without losing aspect ratio
    new_h         : int
                    height, of the image after resizing without losing aspect ratio
    old_w         : int
                    width, of the image before padding
    old_h         : int
                    height, of the image before padding
    padding_w     : int
                    width, of the image after padding
    padding_h     : int
                    height, of the image after padding
    """
    old_h, old_w, c = image.shape
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height

    canvas = np.zeros((height, height, c), np.float32)
    if means is not None:
        canvas[...] = means

    if new_w != old_w or new_h != old_h:
        if interpolation is None:
            image = cv2.resize(image, (new_w, new_h))
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    padding_h = height - new_h
    padding_w = width - new_w

    # parameter for inserting resized image to the middle of canvas
    h_start = max(0, height - new_h - 1) // 2
    w_start = max(0, width - new_w- 1) // 2

    if c > 1:
        canvas[h_start:h_start+new_h, w_start:w_start+new_w] = image
    else:
        if len(image.shape) == 2:
            canvas[h_start:h_start+new_h, w_start:w_start+new_w, 0] = image
        else:
            canvas[h_start:h_start+new_h, w_start:w_start+new_w] = image

    return canvas, new_w, new_h, old_w, old_h, padding_w, padding_h,

def _preprocess(img, input_shape, fill_value=128):
    """
    Preprocess an image for custom YOLOv5 head detector TensorRT model inferencing.

    Parameters
    ----------
    img         : numpy array
                  In BGR format
                  uint8 numpy array of shape (img_h, img_w, 3)
    input_shape : tuple
                  a tuple of (H, W)
    fill_value  : int
                  random values for padding/resizing the image

    Returns
    -------
    img         : numpy array
                  preprocessed image
                  float32 numpy array of shape (3, H, W)
    """ 
    # convert BGR image to RGB image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # resize the image
    img_meta = _aspectaware_resize_padding(image=img, width=input_shape, 
                                           height=input_shape, interpolation=cv2.INTER_CUBIC, means=fill_value)
    img = img_meta[0]/255.0
    img = img.transpose((2, 0, 1)).astype(np.float32)
    # img = img[np.newaxis, ...]

    return img, img_meta[1:]

class postprocess(object):
    """Class for post-processing the outputs from EfficientDet-TensorRT model."""
    def __init__(self, conf_thres, nms_thres, input_size, anchors):
        """
        Initialize parameters, required for postprocessing model outputs.

        Parameters
        ----------
        conf_thres  : int
                      Threshold value for filtering boxes based on confidence scores
        nms_thres   : int
                      Threshold value for performing non-maximum suppresion
        input_size  : int
                      input_size of the model
        anchors     : numpy array
                      per-configured anchors for post-processing outputs
        """
        self.nms_thres  = nms_thres
        self.conf_thres = conf_thres
        self.input_size = input_size

        assert isinstance(anchors, np.ndarray), "Anchors must be in numpy array dtype!"

        # reshape anchors into grids
        self.anchors = np.reshape(anchors, (3, -1, 2)).astype(np.float32)

        # construct anchors_grid
        self.anchor_grid = np.reshape(np.copy(self.anchors), (3, 1, -1, 1, 1, 2))

    @staticmethod
    def _apply_nms(dets, scores, threshold):
        """
        apply non-maxumim suppression

        Parameters
        ----------
        dets      : numpy array
                    array in (num of dets x 4) format
        threshold : numpy array
                    array in (num of dets) format

        Retuens
        -------
        """
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1] # get boxes with more ious first

        keep = []
        while order.size > 0:
            i = order[0] # pick maxmum iou box
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1) # maximum width
            h = np.maximum(0.0, yy2 - yy1 + 1) # maxiumum height
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]

        return keep

    def _clip_boxes(self, predict_boxes, input_size=None):
        """
        Clip the invalid boxes such as
        1. negative values for width and height
        2. values greater than respective width and height

        Parameters
        ----------
        predict_boxes : numpy array
                        numpy array (num_of detection , 4) format
        input_size    : int
                        dimension of input image to the model
        """ 

        # use initialized value in postprocessing if no value is passed
        if input_size is None:
            input_size = self.input_size

        height, width = input_size, input_size
        predict_boxes[np.isnan(predict_boxes)] = 0
        predict_boxes[:, 0][predict_boxes[:, 0] < 0] = 0
        predict_boxes[:, 1][predict_boxes[:, 1] < 0] = 0

        predict_boxes[:, 2][predict_boxes[:, 2] > width]  = (width - 1) 
        predict_boxes[:, 3][predict_boxes[:, 3] > height] = (height - 1)

        return predict_boxes

    @staticmethod
    def _xywh2xyxy(boxes):
        """
        Convert `xywh` boxes to `xyxy` boxes

        Parameters
        ----------
        boxes : numpy array 
                boxes, generated from _constructed function
                (batch_size, n , 5)
        """

        temp_boxes = np.zeros(boxes.shape)
        temp_boxes[:, :, 0] = boxes[:, :, 0] - boxes[:, :, 2] / 2
        temp_boxes[:, :, 1] = boxes[:, :, 1] - boxes[:, :, 3] / 2
        temp_boxes[:, :, 2] = boxes[:, :, 0] + boxes[:, :, 2] / 2
        temp_boxes[:, :, 3] = boxes[:, :, 1] + boxes[:, :, 3] / 2
        boxes[:, :, :4]     = temp_boxes[:, :, :4]
        
        return boxes

    def _construct_boxes(self, outputs, imgsz):
        """
        Construct bounding boxes from TensorRT outputs

        Parameters
        ----------
        outputs : List of numpy arrays
                  List containing np arrays which corresponds to image zoom factor
                  [(batch_size, detection_layer, zoom, zoom, 6)]
        imgsz   : tuple or list
                  Dimensions of input data to model
                  (img_w, img_h)
        """ 
        boxes = []
        for idx, output in enumerate(outputs):
            batch = output.shape[0]
            feature_w = output.shape[2]
            feature_h = output.shape[3]

            # Feature map correspoonds to the original image zoom factor
            stride_w = int(imgsz[0] / feature_w)
            stride_h = int(imgsz[1] / feature_h)

            grid_x, grid_y = np.meshgrid(np.arange(feature_w), np.arange(feature_h))

            # rescale the bounding boxes and swap with pre-configured bounding boxes
            pred_boxes = np.zeros(output[..., :4].shape)
            pred_boxes[..., 0] = (expit(output[..., 0]) * 2.0 - 0.5 + grid_x) * stride_w # cx
            pred_boxes[..., 1] = (expit(output[..., 1]) * 2.0 - 0.5 + grid_y) * stride_h # cy
            pred_boxes[..., 2:4] = (expit(output[..., 2:4]) * 2) ** 2 * self.anchor_grid[idx] # wh

            conf = expit(output[..., 4])
            cls  = expit(output[..., 5])

            # reshape the boxes
            pred_boxes = np.reshape(pred_boxes, (batch, -1, 4))
            conf = np.reshape(conf, (batch, -1, 1))
            cls  = np.reshape(cls, (batch, -1, 1))
                        
            boxes.append(np.concatenate((pred_boxes, conf, cls), axis=-1).astype(np.float32))

        return np.hstack(boxes)

    @staticmethod
    def _scale_coords(img1_shape, coords, img0_shape):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

    
        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        return coords

    def __call__(self, trt_outputs, metas, conf_thres=None, nms_thres=None):
        """
        Apply arrays of post-processing operations to inferred outputs from TensorRT model.
        """
        # use class initialized values for post-processing if no value is passed
        if conf_thres is None:
            conf_thres = self.conf_thres
        if nms_thres is None:
            nms_thres = self.nms_thres

        preds = self._construct_boxes(outputs=trt_outputs, imgsz=(self.input_size, self.input_size))
        preds = self._xywh2xyxy(boxes=preds)

        final_preds = []
        for pred in preds:
            # clip the boxes
            pred = self._clip_boxes(predict_boxes=pred)

            # filter out the detections with scores lower than confidence threshold 
            score_mask = pred[:, 4] >= conf_thres # generate mask

            # filter out the boxes with mask
            pred = pred[score_mask]

            # perform nms
            keep =self._apply_nms(dets=pred[:, :4], scores=pred[:, 4], threshold=nms_thres)

            # calibrated_boxes = pred[keep]
            # calibrated_boxes = self._invert_affine(metas=metas, preds=pred[keep])
            calibrated_boxes = self._scale_coords((self.input_size, self.input_size), pred[keep], (metas[3], metas[2]))

            final_preds.append(calibrated_boxes)

        return final_preds

# TensorRT helper, getter, setter functions
class HostDeviceMem(object):
    """Simple helper data class that's a little nicer to use than a 2-tuple"""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    """Allocates all host/device in/out buffers required for an engine."""

    inputs   = []
    outputs  = []
    bindings = []
    stream   = cuda.Stream()
    for binding in engine:
        size  = trt.volume(engine.get_binding_shape(binding)) * \
                engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # Allocate host and device buffers.
        host_mem   = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes) 
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream):
    """do_inference (for TensorRT 7.0+)

    This function is generalized for multiple inputs/outputs for full
    dimension networks.
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

class YOLOv5HeadModel(object):
    """
    YOLOv5HeadModel is a wrapper class for inferencing the finetuned custom head detector YOLOv5 TensorRT runtime model.
    YOLOv5HeadModel has three variants: small, medium, large based on the depth of feature extractor network.

    YOLOv5 Head Detection Model conversion process
    ----------------------------------------------
    Finetuned custom head detector YOLOv5 model ==> ONNX model ==> customYOLOv5 TensorRT model.

    Further Infos
    -------------
    referenced repo : https://github.com/ultralytics/yolov5
    custom repo : 
    postprocessing : https://colab.research.google.com/drive/1RoxIaslU3QDmb9zNc0xrAmcnCK5vMN_7?usp=sharing#scrollTo=_uPq9mVgiBql
    """

    def _load_engine(self, engine_path):
        TRTbin = engine_path
        with open(TRTbin, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _create_context(self, engine):
        return engine.create_execution_context()

    def __init__(self, engine_path, nms_thres, conf_thres, input_size=1024, anchors=None):
        """
        This is a single-class detection model.
        Initialize the parameters, required for building custom YOLOv5 Head detector- TensorRT model.

        Parameters
        ----------
        engine_path : str
                      Path of custom YOLOv5 TensorRT engine model file
        nms_thres   : int
                      Threshold value for performing non-maximum suppression
        conf_thres  : int
                      Threshold value for filtering the boxes, outputted from the model
        input_size  : int or list
                      Dimension for input data to TensorRT model
        anchors     : numpy array
                      Preconfigured anchors in (no_of_detect_layers, no_of_anchors, 2)
                                               (3, 3, 2)
        
        Attributes
        ----------
        trt_logger    : TensorRT Logger instance
        cuda_ctx      : CUDA context
        postprocessor : Object
                        Collection of postprocessing functions such as non-maximum suppression, clipboxes, scales_coords
        """

        # create a CDUA context, to be used by TensorRT engine
        self.cuda_ctx = cuda.Device(0).make_context() # use GPU:0

        self.engine_path = engine_path if isinstance(engine_path, str) else str(engine_path)

        # check if the engine file exists
        assert ops.isfile(self.engine_path), "YOLOv5 TensorRT Engine file does not exists. Please check the path!"

        # threshold values
        self.nms_thres = nms_thres
        self.conf_thres = conf_thres

        # input_size of model
        self.input_size = int(input_size)

        if anchors is None:
            self.anchors = np.array([[8,9, 19,21, 36,43], [71,86, 114,130, 162,199], [216,255, 295,331, 414,438]])
        else:
            self.anchors = np.array(anchors) if isinstance(anchors, list) else anchors

        # output shapes
        self.output_sizes = [(1, 3, 128, 128, 6), (1, 3, 64, 64, 6), (1, 3, 32, 32, 6)]

        self.postprocess = postprocess(conf_thres=self.conf_thres, nms_thres=self.nms_thres,
                                       input_size=self.input_size, anchors=self.anchors)
        
        # make inference function instance
        self.inference_fn = do_inference
        # setup logger
        self.trt_logger = trt.Logger(trt.Logger.INFO)

        # load engine
        self.engine = self._load_engine(self.engine_path)

        try:
            self.context = self._create_context(self.engine)
            self.inputs, self.outputs, self.bindings, self.stream = \
                allocate_buffers(self.engine)
        except Exception as e:
            self.cuda_ctx.pop()
            del self.cuda_ctx
            raise RuntimeError("Fail to allocate CUDA resources") from e

    def __del__(self):
        """Free CUDA memories"""
        del self.stream
        del self.outputs
        del self.input_size

        # release the memory occupied by cuda context creation
        self.cuda_ctx.pop()
        del self.cuda_ctx

    def detect(self, img):
        """
        Detect heads in the input image.
        Perform inference with custom YOLOv5 head detector TensorRT model.

        Parameters
        ----------
        img : numpy array
              image data in numpy array format

        Returns
        -------

        """

        preprocessed_img, metas = _preprocess(img=img, input_shape=self.input_size,
                                              fill_value=128)

        # set host input to the image. The do_inference() function 
        # will copy the input to the GPU before executing
        self.inputs[0].host = np.ascontiguousarray(preprocessed_img)
        trt_outputs = self.inference_fn(
            context=self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream
        )

        # Since TensorRT gives out flat arrays, reshape into appropriate shapes
        trt_outputs = [np.reshape(output, self.output_sizes[idx]) for idx,output in enumerate(trt_outputs)]
        # Perform postprocessing on TensorRT outputs
        preds = self.postprocess(trt_outputs=trt_outputs, metas=metas)

        return preds

if __name__ == "__main__":
    
    model = YOLOv5HeadModel(engine_path="./checkpoints/head_yolov5_1.trt", nms_thres=0.5, conf_thres=0.3)
    image = cv2.imread("/home/htut/Desktop/naplab/cameras/test/221/221.jpg")
    results = model.detect(image, )
    color = [np.random.randint(0, 255), 0, np.random.randint(0, 255)]
    if results[0].any():
        for box in results[0]:
            x1, y1, x2, y2, _, _ = box
            cv2.rectangle(img=image, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=color, thickness=2)

    cv2.imwrite("TensorRT.jpg", image)
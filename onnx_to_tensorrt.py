from __future__ import print_function

import os
import argparse

import tensorrt as trt

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_file", type=str, required=True, help="ONNX weight file")
    parser.add_argument("--save_dir", type=str, default="checkpoints/", help="Dir to save de-serialized TensorRT runtime model")
    parser.add_argument("-v", "--verbose", action='store_true',
                        help='enable verbose output (for debugging)')
    return parser.parse_args()

EXPLICIT_BATCH = []
if trt.__version__[0] >= '7':
    EXPLICIT_BATCH.append(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

def build_engine(onnx_file_path, engine_file_path, batch_size=1, verbose=False, FP16=True):
    """Take an ONNX file and creates a TensorRT engine"""
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 28
        builder.max_batch_size = batch_size
        builder.fp16_mode = True
        # builder.strict_type_contraints = True

        # Parse  model file
        print("Loading ONNX file from path {}...".format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        if trt.__version__[0] == '7':
            # The actual head*.onnx may be generated with a different batch size
            # Reshape the input to batch size 1

            shape = list(network.get_input(0).shape)
            print(shape)
            shape[0] = int(batch_size)
            network.get_input(0).shape = shape
        
        print("Completed parsing of ONNX file")
        print('Building an engine; this may take a while...')
        engine = builder.build_cuda_engine(network)
        print(engine)
        print('Completed creating engine')
        with open(engine_file_path, 'wb') as f:
            f.write(engine.serialize())
        return engine

if __name__ == "__main__":
    args = parser_args()
    onnx_file = args.onnx_file
    save_dir = os.path.abspath(args.save_dir)

    if not os.path.exists(onnx_file):
        raise SystemExit('ERROR: file (%s) not found!  You might want to run yolo_to_onnx.py first to generate it.' % onnx_file)
    if not os.path.isdir(args.save_dir):
        raise SystemExit("ERROR: Dir `{:s}` does not exists.".format(save_dir))

    # Example : onnx_file - custom_head_YOLOv5_4.onnx
    batch_size = int(list(onnx_file.split('.')[0])[-1])
    trt_engine = save_dir + "/head_yolov5_{}.trt".format(batch_size)
    _ = build_engine(onnx_file_path=onnx_file, engine_file_path=trt_engine, batch_size=batch_size)
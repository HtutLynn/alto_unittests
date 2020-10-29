#!/bin/bash

set -e

echo "Downloading onnx model file for custom YOLOv5 head models..."
wget https://altotechpublic.s3-ap-southeast-1.amazonaws.com/naplabchula/models/custom_head_YOLOv5_1.onnx -q --show-progress --no-clobber
wget https://altotechpublic.s3-ap-southeast-1.amazonaws.com/naplabchula/models/custom_head_YOLOv5_4.onnx -q --show-progress --no-clobber

echo "Done!"
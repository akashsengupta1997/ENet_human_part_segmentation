# ENet_human_part_segmentation
Quick and dirty human part segmentation code using ENet using Kereas 2.1.0, Tensorflow 1.12.0, opencv-python 3.4.3, matplotlib 3.0.2.


Use realtime_demo.py for a real time demonstration with attached webcam/laptop camera. Edit the path to the weights in the file (when using 'load_model'). The provided models all use 256x256x3 inputs (hence, set img_wh to 256) and give either 64x64x1 outputs or 256x256x1 outputs, depending on whether upsample was enabled or not. Set img_dec_wh to 64 or 256 accordingly. 

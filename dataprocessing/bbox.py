import sys

sys.path.append('.')
import os 
import glob
import numpy as np
from timeit import default_timer as timer
import igl
import traceback
import math
import torch
import pytorch_kinematics as pk
import math

path = 'datasets/arm/UR5/meshes/collision/'
input_file_list = ['forearm','shoulder','upper_arm','wrist_1','wrist_2','wrist_3']

numsamples = 1000000

for input_file in input_file_list:
    v, f = igl.read_triangle_mesh(path+input_file+'.obj')
    #v = 0.3*v
    vmax = np.max(v, axis=0)
    vmin = np.min(v, axis=0)
    bbox = np.concatenate((vmax,vmin),axis=0)
    print(bbox)
    np.save('{}/'.format(path)+input_file+'bbox',bbox)


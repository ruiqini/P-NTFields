import sys
sys.path.append('.')
from models import model_res_sigmoid_multi as md
from os import path

modelPath = './Experiments/Gib_multi'

dataPath = './datasets/gibson/'

model    = md.Model(modelPath, dataPath, 3, 2, device='cuda:0')

model.train()



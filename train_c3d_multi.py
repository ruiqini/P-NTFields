from models import model_res_sigmoid_multi as md
#from models import model_mlp_linear as md
from os import path

#ToyProblem_BlockGibsonObsDisModel
#ToyProblem_BlockC3DObsDisModel
#modelPath = './Experiments/Gib_cabin'         #ToyProblem_BlockObsDisModel
modelPath = './Experiments/C3D_multi'
#modelPath = './Experiments/Gib_cabin_linear'
#modelPath = './Experiments/Gib_res_changelr_scale'

dataPath = './datasets/c3d/'#Arona,Cabin,Bolton

model    = md.Model(modelPath, dataPath, 3, 8, device='cuda:0')

#model.load('./Experiments/Model_Epoch_00200_ValLoss_0.008947935368501603.pt')

model.train()



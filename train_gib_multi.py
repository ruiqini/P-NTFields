from models import model_res_sigmoid_multi as md
#from models import model_mlp_linear as md
from os import path

#ToyProblem_BlockGibsonObsDisModel
#ToyProblem_BlockC3DObsDisModel
#modelPath = './Experiments/Gib_cabin'         #ToyProblem_BlockObsDisModel
#modelPath = './Experiments/Gib_cabin_fixlr'
modelPath = './Experiments/Gib_multi'#arona,bolton,cabin,A_test
#modelPath = './Experiments/Gib_res_changelr_scale'

dataPath = './datasets/gibson/'#Arona,Cabin,Bolton

model    = md.Model(modelPath, dataPath, 3, 2, device='cuda:0')

#model.load('./Experiments/Model_Epoch_00200_ValLoss_0.008947935368501603.pt')

model.train()



from models import model_res_sigmoid as md
from os import path

#ToyProblem_BlockGibsonObsDisModel
#ToyProblem_BlockC3DObsDisModel
modelPath = './Experiments/Arm'         #ToyProblem_BlockObsDisModel
dataPath = './datasets/arm/UR5'

model    = md.Model(modelPath, dataPath, 6, device='cuda')

#model.load('./Experiments/Model_Epoch_00200_ValLoss_0.008947935368501603.pt')

model.train()
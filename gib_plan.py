from models import model_res_sigmoid_multi as md
import torch
import os 
import numpy as np
import matplotlib.pylab as plt
import torch
from torch import Tensor
from torch.autograd import Variable, grad

from timeit import default_timer as timer
import math
import igl
import open3d as o3d


modelPath = './Experiments/Gib'

dataPath = './datasets/gibson/'
womodel    = md.Model(modelPath, dataPath, 3, 2, device='cuda')


womodel.load('./Experiments/Gib_multi/Model_Epoch_10000_ValLoss_1.221157e-01.pt')#
womodel.network.eval()
    
max_x = 0
max_y = 0
max_z = 0
#for gib_id in range(2):
gib_id = 0
v, f = igl.read_triangle_mesh("datasets/gibson/"+str(gib_id)+"/mesh_z_up_scaled.off")        
print(gib_id)

vertices=v*20
faces=f

vertices = torch.tensor(vertices, dtype=torch.float32, device='cuda')
faces = torch.tensor(faces, dtype=torch.long, device='cuda')
triangles = vertices[faces].unsqueeze(dim=0)

B = np.load("datasets/gibson/"+str(gib_id)+"/B.npy")
B = Variable(Tensor(B)).to('cuda')


for ii in range(5):
    start_goal = np.array([[-6,-7,-6,2,7,-2.5]])
    XP=start_goal
    XP = Variable(Tensor(XP)).to('cuda')
    XP=XP/20.0

    dis=torch.norm(XP[:,3:6]-XP[:,0:3])


    start = timer()

    point0=[]
    point1=[]

    point0.append(XP[:,0:3].clone())
    point1.append(XP[:,3:6].clone())
    #print(id)

    iter=0
    while dis>0.06:
        gradient = womodel.Gradient(XP.clone(), B)
   
        XP = XP + 0.03 * gradient
        dis=torch.norm(XP[:,3:6]-XP[:,0:3])
        #print(XP)
        point0.append(XP[:,0:3].clone())
        point1.append(XP[:,3:6].clone())
        iter=iter+1
        if(iter>500):
            break
    #point0.append(p[:,3:6][0])

    end = timer()
    print("plan",end - start)
point1.reverse()
point=point0+point1

xyz= torch.cat(point).to('cpu').data.numpy()#np.asarray(point)

xyz=20*xyz

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(xyz)

mesh = o3d.io.read_triangle_mesh("datasets/gibson/"+str(gib_id)+"/mesh_z_up_scaled.off")
        
mesh.scale(20, center=(0,0,0))

mesh.compute_vertex_normals()

o3d.visualization.draw_geometries([mesh,pcd])



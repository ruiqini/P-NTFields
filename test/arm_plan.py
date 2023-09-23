import sys

sys.path.append('.')
import torch
import pytorch_kinematics as pk
import numpy as np

import matplotlib
import matplotlib.pylab as plt
import sys

sys.path.append('.')
import open3d as o3d
#model_res_sigmoid
#from models import model_res_sigmoid_nt as md 
from models import model_res_sigmoid as md 

from timeit import default_timer as timer
import igl

def Arm_FK(sampled_points, out_path_ ,path_name_,end_effect_):
    shape=sampled_points.shape
    pointsize = 0
    
    d = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    chain = pk.build_serial_chain_from_urdf(
        open(out_path_+'/'+path_name_+".urdf").read(), end_effect_)
    chain = chain.to(dtype=dtype, device=d)

    scale = np.pi/0.5
    
    th_batch = torch.tensor(scale*sampled_points, requires_grad=True).cuda()

    tg_batch = chain.forward_kinematics(th_batch, end_only = False)

    p_list=[]
    iter = 0
    pointsize = 0

    traj = []

    mesh_list = []
    for tg in tg_batch:
        if iter>1:

            mesh = o3d.io.read_triangle_mesh(out_path_+'/meshes/visual/'+tg.replace('_link','')+'.obj')
            
            mesh_list.append(mesh)
            v = np.asarray(mesh.vertices)
            #print(v.shape[0])
            nv = np.ones((v.shape[0],4))
            pointsize = pointsize+v.shape[0]

            nv[:,:3] = v
            m = tg_batch[tg].get_matrix()
            t=torch.from_numpy(nv).cuda()
            p=torch.matmul(m[:],t.T)
            #p=p.cpu().numpy()
            p=torch.permute(p,(0,2,1))
            p_list.append(p)
            
            del m,p,t,nv, v
        iter=iter+1

    wholemesh = o3d.geometry.TriangleMesh()

    for ii in range(len(mesh_list)):
        #ii=-1
        mesh = mesh_list[ii]
        p = p_list[ii].detach().cpu().numpy()
        #print(p.shape)
        for jj in range(len(p)):
            pp = p[jj]
            mesh.vertices = o3d.utility.Vector3dVector(pp[:,:3])
            wholemesh+=mesh
    #'''
    #o3d.io.write_triangle_mesh('arm.obj', wholemesh)
    wholemesh.compute_vertex_normals()
    #print(pointsize)
    return wholemesh

#base = np.array([[0, -0.5*np.pi, 0.0, -0.5*np.pi,0.0,0.0]])

modelPath = './Experiments/UR5'         
dataPath = './datasets/arm/UR5'

model    = md.Model(modelPath, dataPath, 6, device='cuda')


model.load('./Experiments/UR5/Model_Epoch_10000_ValLoss_3.526511e-03.pt')

for ii in range(10):
    
    XP=torch.tensor([[0.00,0.0,0.0,-0.00,0.00,-0.00,
                        -1.3, 0.4, 1.1, 0.5,-0.5,0.0]]).cuda()
    XP=torch.tensor([[-2.2, 0.4, 1.1, 0.5,-0.5,0.9,
                        -1.3, 0.4, 1.1, 0.5,-0.5,0.0]]).cuda()
    
    #XP=torch.tensor([[-1.3, 0.4, 1.1, 0.2,-1.5,0.9,
    #                -2.5, 0.8, 1.4, 1.2,0.5,0.1]]).cuda()
    
    BASE=torch.tensor([[0, -0.5*np.pi, 0.0, -0.5*np.pi,0.0,0.0,
                        0, -0.5*np.pi, 0.0, -0.5*np.pi,0.0,0.0]]).cuda()
    
    XP = XP+BASE

    scale = np.pi/0.5
    XP=XP/scale

    dis=torch.norm(XP[:,6:]-XP[:,:6])

    #model.plot(10,10,1.2)

    point0= []#torch.tensor(()).cuda()
    point1= []#torch.tensor(()).cuda()
    #p=XP.to('cpu').data.numpy()
    #print(XP[:,0:3])
    point0.append(XP[:,:6])
    point1.append(XP[:,6:])
    start = timer()
    time1 = 0
    iter=0
    while dis>0.03:
        #start1 = timer()
        #tau, Xp = model.network(XP)
        
        
        gradient = model.Gradient(XP.clone())
        
        XP = XP + 0.015 * gradient
        dis=torch.norm(XP[:,6:]-XP[:,:6])
        #print(XP)
        #p=XP.to('cpu').data.numpy()
        point0.append(XP[:,:6])
        point1.append(XP[:,6:])
        
        iter=iter+1
        if iter>300:
            break
    
    end1 = timer()
    #print(iter)

    print("plan",end1 - start)
    #print("time1",time1)
    point1.reverse()
    point=point0+point1
    xyz=torch.cat(point).to('cpu').data.numpy()#np.asarray(point)
#xyz=xyz*scale
#print(xyz)
xyz0=np.zeros((2,6))
xyz0[0,:]=xyz[0,:]
xyz0[1,:]=xyz[-1,:]

#b = 28
scale = np.pi/0.5

wholemesh = Arm_FK(xyz[0::1,:],'datasets/arm/UR5','UR5','wrist_3_link')

def length(path):
        size=path.shape[0]
        l=0
        for i in range(size-1):
            l+=np.linalg.norm(path[i+1,:]-path[i,:])
        return l
print(length(xyz))


file_path = 'datasets/arm/'
mesh_name = 'untitled_scaled.off'

path = file_path + 'UR5'+'/'

#dirname = os.path.dirname(f)
obstacle = o3d.io.read_triangle_mesh(path + mesh_name)
vertices=np.asarray(obstacle.vertices)
faces=np.asarray(obstacle.triangles)
obstacle.vertices = o3d.utility.Vector3dVector(vertices)


obstacle.compute_vertex_normals()
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.1, origin=[0, 0, 0])

o3d.visualization.draw_geometries([obstacle,wholemesh,mesh_frame])


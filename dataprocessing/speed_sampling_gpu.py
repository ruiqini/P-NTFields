import os 
import glob
import numpy as np
from timeit import default_timer as timer
import igl
import traceback
import math
import torch
import pytorch_kinematics as pk

import bvh_distance_queries
import math
import matplotlib.pyplot as plt


def bbox_corner(bbox):
    corner = torch.ones((4,8)).cuda()
    iter=0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                corner[0,iter]=bbox[0+i*3]
                corner[1,iter]=bbox[1+j*3]
                corner[2,iter]=bbox[2+k*3]
                iter+=1

    return corner

def bbox_axis(bbox):
    axis = torch.zeros((3,3)).cuda()
    axis[0,0]=1#bbox[0]-bbox[3]
    axis[1,1]=1#bbox[1]-bbox[4]
    axis[2,2]=1#bbox[2]-bbox[5]

    return axis

def cross_matrix(transform_row):
    matrix = torch.zeros((transform_row.shape[0],3,3)).cuda()
    
    a = transform_row[:,0]
    b = transform_row[:,1]
    c = transform_row[:,2]

    l0 = torch.clamp(torch.sqrt(c**2+b**2), min=1e-6)
    l1 = torch.clamp(torch.sqrt(c**2+a**2), min=1e-6)
    l2 = torch.clamp(torch.sqrt(b**2+a**2), min=1e-6)

    matrix[:,0,1] = c/l0
    matrix[:,0,2] = -b/l0
    matrix[:,1,0] = -c/l1
    matrix[:,1,2] = a/l1
    matrix[:,2,0] = b/l2
    matrix[:,2,1] = -a/l2
    return matrix

def separate_axis(transform):
    axis = torch.zeros((transform.shape[0],15,3)).cuda()
    #axis[0,0]=1#bbox[0]-bbox[3]
    #axis[1,1]=1#bbox[1]-bbox[4]
    #axis[2,2]=1#bbox[2]-bbox[5]
    axis[:,0:3,:] = torch.eye(3).cuda()
    axis[:,3:6,:] = transform[:,0:3,0:3]
    axis[:,6:9,:]   = cross_matrix(transform[:,0,0:3])
    axis[:,9:12,:]  = cross_matrix(transform[:,1,0:3])
    axis[:,12:15,:] = cross_matrix(transform[:,2,0:3])

    return axis

def obb_collision(rob_corner, obs_corner, transform, margin):

    return 0


def arm_obstacle_obb(th_batch, chain, out_path_, margin):
    whole_dis = []
    end_dis = []
    batch_size = 2000
    scale = 5.0
    modelPath = './Experiments/UR5_SDF'#arona,bolton,cabin,A_test
#modelPath = './Experiments/Gib_res_changelr_scale'

    dataPath = './datasets/arm/'#Arona,Cabin,Bolton#filePath = './Experiments/Gibson'
    model_list = []
    bbox_list = []
    rob_corner_list = []
    input_file_list = ['upper_arm','forearm','wrist_1','wrist_2','wrist_3']
    for input_file in input_file_list:
        bbox = np.load('./datasets/arm/UR5/meshes/collision/'+input_file+'bbox.npy')
        bbox = torch.tensor(bbox).cuda().float()
        bbox[:3] = bbox[:3] + margin
        bbox[3:] = bbox[3:] - margin
        rob_corner_list.append(bbox_corner(bbox))
    #hard code bbox of obs, can be further accelerated by BVH
    bbox = torch.tensor([0.7,-0.5,1.0,-0.6,-0.7,-0.7]).cuda().float()
    obs_points = bbox_corner(bbox)

    batch_size = 50000

    where_list = []
    for batch_id in range(math.floor(th_batch.shape[0]/batch_size)+1):
        if batch_id*batch_size==th_batch.shape[0]:
            break
        #print(batch_id)
        local_th_batch = th_batch[batch_id*batch_size:min((batch_id+1)*batch_size,th_batch.shape[0]),:]
        tg_batch = chain.forward_kinematics(
                    local_th_batch
                    , end_only = False)

        p_list=[]
        iter = 0
        p_size_list = []
        
        where_coll = torch.zeros(local_th_batch.shape[0], dtype=torch.bool).cuda()

        for tg in tg_batch:
            if iter>2:
                m = tg_batch[tg].get_matrix()
                rob_points= m@rob_corner_list[iter-3]
                #print(iter)
                axis = separate_axis(m)
                rob_axis_point = axis@rob_points[:,0:3,:]
                #print(rob_axis_point.shape)
                #print(obs_points.shape)
                obs_axis_point = axis@obs_points[0:3,:]

                rob_axis_point_min,_ = torch.min(rob_axis_point,dim=2)
                rob_axis_point_max,_ = torch.max(rob_axis_point,dim=2)
                
                obs_axis_point_min,_ = torch.min(obs_axis_point,dim=2)
                obs_axis_point_max,_ = torch.max(obs_axis_point,dim=2)

                where0 = rob_axis_point_max<obs_axis_point_min
                where1 = obs_axis_point_max<rob_axis_point_min

                where = torch.cat((where0,where1),dim=1)
                #print(where.shape)
                nonsep = torch.all((where == False),dim=1)
                #print(nonsep.shape)
                combine = torch.cat((nonsep.unsqueeze(1),where_coll.unsqueeze(1)),dim=1)
                where_coll = torch.any(combine == True,dim=1)

                del m
            iter = iter+1
        where_list.append(nonsep)
        #print(where_coll.shape)
    return torch.cat(where_list,0)

def arm_obstacle_distance(th_batch, chain, out_path_, triangles_obs):
    whole_dis = []
    end_dis = []
    batch_size = 50000
    for batch_id in range(math.floor(th_batch.shape[0]/batch_size)+1):
        if batch_id*batch_size==th_batch.shape[0]:
            break
        #print(batch_id)
        tg_batch = chain.forward_kinematics(
            th_batch[batch_id*batch_size:
                    min((batch_id+1)*batch_size,th_batch.shape[0]),:]
                    , end_only = False)

        p_list=[]
        iter = 0
        p_size_list = []
        
        for tg in tg_batch:
            if iter>1:
                #print(iter)
                v, f = igl.read_triangle_mesh(out_path_+'/meshes/collision/'+tg+'.obj')
                nv = np.ones((v.shape[0],4))
                #pointsize = pointsize+v.shape[0]
                p_size_list.append(v.shape[0])
                nv[:,:3]=v
                m = tg_batch[tg].get_matrix()
                #print(m.shape)
                t=torch.from_numpy(nv).float().cuda()
                p=torch.matmul(m[:],t.T)
                #p=p.cpu().numpy()
                p = torch.permute(p, (0, 2, 1)).contiguous()
                #p=np.transpose(p,(0,2,1))
                p_list.append(p)
                del m,p,t,nv, v
            iter = iter+1
        pointsize = sum(p_size_list)
        #print(pointsize)
        #p = np.concatenate(p_list,axis=1)
        p = torch.cat(p_list, dim=1)
        p = torch.reshape(p,(p.shape[0]*p.shape[1],p.shape[2])).contiguous()
        query_points = p[:,0:3].contiguous()
        query_points = query_points.unsqueeze(dim=0)
        
        bvh = bvh_distance_queries.BVH()

        torch.cuda.synchronize()
        torch.cuda.synchronize()
        distance, closest_points, closest_faces, closest_bcs= bvh(triangles_obs, query_points)
        torch.cuda.synchronize()
        #unsigned_distance = abs()
        #print(distance.shape)
        distance = torch.sqrt(distance).squeeze()
        distance = torch.reshape(distance, (-1, pointsize))
        whole_distance,_ = torch.min(distance, dim=1)
        #distance = distance.detach().cpu().numpy()


        #print(whole_distance)
        whole_dis.append(whole_distance)
        del p, p_list, tg_batch, distance, query_points, bvh,whole_distance
    #unsigned_distance = np.concatenate(whole_p, axis=0)
    unsigned_distance = torch.cat(whole_dis, dim=0)
    #print(unsigned_distance)
    return unsigned_distance

def arm_append_list(X_list, Y_list, chain, out_path_, 
                    triangles_obs,
                    numsamples, dim, offset, margin):
    
    OutsideSize = numsamples + 2
    WholeSize = 0

    scale = math.pi/0.5

    while OutsideSize > 0:
        P  = torch.rand((5*numsamples,dim),dtype=torch.float32, device='cuda')-0.5
        dP = torch.rand((5*numsamples,dim),dtype=torch.float32, device='cuda')-0.5
        rL = (torch.rand((5*numsamples,1),dtype=torch.float32, device='cuda'))*np.sqrt(3)
        nP = P + torch.nn.functional.normalize(dP,dim=1)*rL
        #nP = torch.rand((8*numsamples,dim),dtype=torch.float32, device='cuda')-0.5
        
        PointsInside = torch.all((nP <= 0.5),dim=1) & torch.all((nP >= -0.5),dim=1)
        

        x0 = P[PointsInside,:]
        x1 = nP[PointsInside,:]

        #print(x0.shape[0])
        if(x0.shape[0]<=1):
            continue
        
        th_batch = scale*x0
        print(th_batch.shape)
        where = arm_obstacle_obb(th_batch, chain, out_path_, margin)#.cuda()
        x0 = x0[where]
        x1 = x1[where]
        #print(x0)

        th_batch = scale*x0
        print(th_batch.shape)
        obs_distance0 = arm_obstacle_distance(th_batch, chain, out_path_, triangles_obs)
        print(torch.min(obs_distance0))
        print(torch.max(obs_distance0))
        where_d          =  (obs_distance0 >= offset) & (obs_distance0 <= margin)
        x0 = x0[where_d]
        x1 = x1[where_d]
        y0 = obs_distance0[where_d]

        th_batch = scale*x1
        obs_distance1 = arm_obstacle_distance(th_batch, chain, out_path_, triangles_obs)

        y1 = obs_distance1
        
        print(x0.shape)
        #print(x1.shape)
        #print(y0.shape)
        #print(y1.shape)

        x = torch.cat((x0,x1),1)
        y = torch.cat((y0.unsqueeze(1),y1.unsqueeze(1)),1)

        X_list.append(x)
        Y_list.append(y)
        
        OutsideSize = OutsideSize - x.shape[0]
        WholeSize = WholeSize + x.shape[0]
        
        if(WholeSize > numsamples):
            break
    return X_list, Y_list

def arm_rand_sample_bound_points(numsamples, dim, 
                                 v_obs, f_obs, offset, margin,
                                 out_path_ , path_name_, end_effect_):
    numsamples = int(numsamples)

    d = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    chain = pk.build_serial_chain_from_urdf(
        open(out_path_+'/'+path_name_+".urdf").read(), end_effect_)
    chain = chain.to(dtype=dtype, device=d)

    scale = math.pi/0.5

    v_obs = torch.tensor(v_obs, dtype=torch.float32, device='cuda')
    f_obs = torch.tensor(f_obs, dtype=torch.long, device='cuda')
    t_obs = v_obs[f_obs].unsqueeze(dim=0)

    X_list = []
    Y_list = []

    X_list, Y_list = arm_append_list(X_list, Y_list, chain, out_path_, 
                                    t_obs, numsamples, dim, offset, margin)
  
    X = torch.cat(X_list,0)[:numsamples]
    Y = torch.cat(Y_list,0)[:numsamples]    
    
    sampled_points = X.detach().cpu().numpy()
    distance = Y.detach().cpu().numpy()
    
    distance0 = distance[:,0]
    distance1 = distance[:,1]
    speed  = np.zeros((distance.shape[0],2))
    speed[:,0] = np.clip(distance0 , a_min = offset, a_max = margin)/margin
    speed[:,1] = np.clip(distance1 , a_min = offset, a_max = margin)/margin

    return sampled_points, speed

def point_obstacle_distance(query_points, triangles_obs):
    query_points = query_points.unsqueeze(dim=0)
    #print(query_points.shape)
    bvh = bvh_distance_queries.BVH()
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    distances, closest_points, closest_faces, closest_bcs= bvh(triangles_obs, query_points)
    torch.cuda.synchronize()
    unsigned_distance = torch.sqrt(distances).squeeze()
    #print(closest_points.shape)
    return unsigned_distance

def point_append_list(X_list,Y_list, 
                      triangles_obs, numsamples, dim, offset, margin):
    
    OutsideSize = numsamples + 2
    WholeSize = 0

    while OutsideSize > 0:
        P  = torch.rand((8*numsamples,dim),dtype=torch.float32, device='cuda')-0.5
        dP = torch.rand((8*numsamples,dim),dtype=torch.float32, device='cuda')-0.5
        rL = (torch.rand((8*numsamples,1),dtype=torch.float32, device='cuda'))*np.sqrt(dim)
        nP = P + torch.nn.functional.normalize(dP,dim=1)*rL
        #nP = torch.rand((8*numsamples,dim),dtype=torch.float32, device='cuda')-0.5
        
        PointsInside = torch.all((nP <= 0.5),dim=1) & torch.all((nP >= -0.5),dim=1)
        

        x0 = P[PointsInside,:]
        x1 = nP[PointsInside,:]

        #print(x0.shape[0])
        if(x0.shape[0]<=1):
            continue
        #print(len(PointsOutside))
        

        obs_distance0 = point_obstacle_distance(x0, triangles_obs)
        where_d          =  (obs_distance0 > offset) & (obs_distance0 < margin)
        x0 = x0[where_d]
        x1 = x1[where_d]
        y0 = obs_distance0[where_d]

        obs_distance1 = point_obstacle_distance(x1, triangles_obs)
        
        y1 = obs_distance1

        print(x0.shape)
        #print(x1.shape)
        #print(y0.shape)
        #print(y1.shape)

        x = torch.cat((x0,x1),1)
        y = torch.cat((y0.unsqueeze(1),y1.unsqueeze(1)),1)

        X_list.append(x)
        Y_list.append(y)
        
        OutsideSize = OutsideSize - x.shape[0]
        WholeSize = WholeSize + x.shape[0]
        
        if(WholeSize > numsamples):
            break
    return X_list, Y_list


def point_rand_sample_bound_points(numsamples, dim, 
                                    v_obs, f_obs, offset, margin):
    numsamples = int(numsamples)

    v_obs = torch.tensor(v_obs, dtype=torch.float32, device='cuda')
    f_obs = torch.tensor(f_obs, dtype=torch.long, device='cuda')
    t_obs = v_obs[f_obs].unsqueeze(dim=0)
    
    X_list = []
    Y_list = []
    #N_list = []
    
    X_list, Y_list = point_append_list(X_list,Y_list,  t_obs, numsamples, dim, offset, margin)
   
    X = torch.cat(X_list,0)[:numsamples]
    Y = torch.cat(Y_list,0)[:numsamples]

    sampled_points = X.detach().cpu().numpy()
    distance = Y.detach().cpu().numpy()
    #normal = N.detach().cpu().numpy()
    
    distance0 = distance[:,0]
    distance1 = distance[:,1]
    speed  = np.zeros((distance.shape[0],2))
    speed[:,0] = np.clip(distance0 , a_min = offset, a_max = margin)/margin
    speed[:,1] = np.clip(distance1 , a_min = offset, a_max = margin)/margin
    
    return sampled_points, speed

def sample_speed(path, numsamples, dim):
    
    try:

        global out_path
        out_path = os.path.dirname(path)
        #print(out_path)
        global path_name 
        path_name = os.path.splitext(os.path.basename(out_path))[0]
        print('pp',path)
        global task_name 
        task_name = out_path.split('/')[2]#os.path.splitext(os.path.basename(out_path),'/')
        if task_name=='arm':
            #dim = np.loadtxt(out_path+'/dim')
            global end_effect
            with open(out_path+'/dim') as f:
                iter = 0
                for line in f:
                    data = line.split()
                    if iter==0:
                        dim = int(data[0])
                    else:
                        end_effect = data[0]
                        print(end_effect)
                    iter=iter+1
        file_name = os.path.splitext(os.path.basename(path))[0]
        input_file = os.path.join(out_path,file_name + '_scaled.off')
        out_file = out_path + '/sampled_points.npy'

        print(input_file)
        if os.path.exists(out_file):
            print(f'Exists: {out_file}')
            #return
   
        #out_file = out_path + '/boundary_{}_samples.npz'.format( sigma)


        limit = 0.5
        xmin=[-limit]*dim
        xmax=[limit]*dim
        velocity_max = 1
        
        if task_name=='c3d' or task_name=='test':
            margin = limit/5.0
            offset = margin/10.0 
        elif task_name=='gibson':
            margin = limit/12.0
            offset = margin/10.0 
        else:
            margin = limit/12.0
            offset = margin/10.0 

        v, f = igl.read_triangle_mesh(input_file)

        start = timer()
        if task_name=='arm':
            sampled_points, speed = arm_rand_sample_bound_points(numsamples, dim, 
                    v, f, offset, margin, out_path ,path_name,end_effect)
        else:
            sampled_points, speed = point_rand_sample_bound_points(numsamples, dim, 
                    v, f, offset, margin)

        end = timer()

        print(end-start)

        B = np.random.normal(0, 1, size=(3, 128))

        np.save('{}/sampled_points'.format(out_path),sampled_points)
        np.save('{}/speed'.format(out_path),speed)
        np.save('{}/B'.format(out_path),B)
    except Exception as err:
        print('Error with {}: {}'.format(path, traceback.format_exc()))
    

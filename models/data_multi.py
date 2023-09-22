import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable, grad


class Database(torch.utils.data.Dataset):
    def __init__(self, path, device, len):
        # Creating identical pairs
        self.device = device
        self.path = path
        self.len = len


    def __getitem__(self, index):
        #points = self.points[index]
        #speed = self.speed[index]
        path = self.path + str(index)
        points = np.load('{}/sampled_points.npy'.format(path)).astype(np.float16)
        speed = np.load('{}/speed.npy'.format(path))
        B = np.load('{}/B.npy'.format(path))

        points = Variable(Tensor(points))
        speed  = Variable(Tensor(speed))

        B = Variable(Tensor(B))
        #print(points.shape)
        #print(speed.shape)
        data = torch.cat((points,speed),dim=1)
        #grid = self.grid
        return data, B, index
    def __len__(self):
        return self.len#2,8




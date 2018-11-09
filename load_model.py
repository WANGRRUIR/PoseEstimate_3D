import torch.utils.data.dataloader
from models.pose_estimate_3D import StackedHourgalss3D
from torch.autograd import Variable as Variable
from data.handle.mpii import MPII
from utils.utils import AverageMeter,adjust_learning_rate
from utils.eval import Accuracy,MPJPE
import ref
import os
from data.handle.fusion import  Fusion
from models.FusionCriterion import FusionCriterion
from opts import opts
import numpy as np
for i in range(3):
    print(i)

opt=opts().parse()
train_loador=torch.utils.data.DataLoader(Fusion(opt,'train'),batch_size=opt.trainBatch,shuffle=True,num_workers=int(ref.nThreads))

dataiter=iter(train_loador)
input,target2D,target3D,meta=dataiter.next()
for i in range(64):
    for j in range(64):
        if(target2D[0][0][i][j]!=0):
            print(target2D[0][0][i][j],i,j)





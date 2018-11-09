import torch.utils.data.dataloader
from models.pose_estimate_3D import StackedHourgalss3D
from utils.utils import AverageMeter,adjust_learning_rate
from utils.eval import Accuracy,MPJPE
import ref
import os
from data.handle.fusion import Fusion
from models.FusionCriterion import FusionCriterion
from opts import opts

#opt_gpu='0'
#os.environ["CUDA_VISIBLE_DEVICES"]=opt_gpu

opt=opts().parse()
train_loador=torch.utils.data.DataLoader(Fusion(opt,'train'),batch_size=opt.trainBatch,shuffle=True,num_workers=int(ref.nThreads))
if opt.loadModel!='none':
    model=torch.load(opt.loadModel)
else:
    model=StackedHourgalss3D(opt.nStack, opt.nModules, opt.nFeats, opt.nRegModules)
model=model.cuda()

criterion=torch.nn.MSELoss()
criterion=criterion.cuda()
optimizer=torch.optim.RMSprop(
    model.parameters(),lr=opt.LR,alpha=ref.alpha,eps=ref.epsilon,weight_decay=ref.weightDecay,momentum=ref.momentum
)
criterion=criterion.cuda()



for epoch in range(1,opt.nEpochs+1):
    model.train()
    Loss, Acc, Mpjpe, Loss3D = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    for i,(input,target2D,target3D,meta) in enumerate(train_loador):
        inputs=input.float().cuda()
        targets_2D=target2D.float().cuda()
        targets_3D=target3D.float().cuda()
        output=model(inputs)

        reg=output[opt.nStack]

        loss=FusionCriterion(opt.regWeight,opt.varWeight)(reg,targets_3D)
        Loss3D.update(loss.item(),input.size(0))

        for j in range(1,opt.nStack):
            loss+=criterion(output[j],targets_2D)

        Loss.update(loss.item(), input.size(0))
        acc=Accuracy((output[opt.nStack - 1].data).cpu().numpy(), (targets_2D.data).cpu().numpy())
        Acc.update(acc)
        mpjpe,num3D=MPJPE((output[opt.nStack - 1].data).cpu().numpy(), (reg.data).cpu().numpy(),meta)
        if num3D>0:
            Mpjpe.update(mpjpe,num3D)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(epoch,i,loss.data,acc)

    if epoch % opt.valIntervals == 0:
        torch.save(model, os.path.join(opt.saveDir, 'model_{}.pth'.format(epoch)))
    adjust_learning_rate(optimizer,epoch,opt.dropLR,opt.LR)

    print('Loss:',Loss.avg,'Acc:',Acc.avg,'Mpjpe:',Mpjpe.avg,'Loss3D',Loss3D.avg)


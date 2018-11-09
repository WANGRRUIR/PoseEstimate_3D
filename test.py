import torch.utils.data
from opts import opts
from data.handle.fusionvalue import Fusion
from utils.eval import Accuracy, getPreds, MPJPE
from utils.debugger import Debugger
from utils.eval import getPreds
import numpy as np



opt = opts().parse()

train_loader = torch.utils.data.DataLoader(
      Fusion(opt, 'train'), 
      batch_size = 1)

model = torch.load('model_10.pth',map_location=lambda storage,loc: storage)
criterion = torch.nn.MSELoss()


num_epoch = 1

for epoch in range(num_epoch):

	for i, (input1, target2D, target3D, meta) in enumerate(train_loader):

		input_var = torch.autograd.Variable(input1).float()
		target2D_var = torch.autograd.Variable(target2D).float()
		target3D_var = torch.autograd.Variable(target3D).float()

		print(target2D_var,target3D_var)

		output = model(input_var)
		print(output)

		pred = getPreds((output[-2].data).cpu().numpy())[0] * 4
		reg = (output[-1].data).cpu().numpy().reshape(pred.shape[0], 1)

		print(pred,(reg + 1) / 2. * 256)








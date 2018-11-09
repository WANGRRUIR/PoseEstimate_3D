import torch
from utils.debugger import Debugger
from utils.eval import getPreds
import cv2
import numpy as np
from PIL import Image

def process_image(filename, mwidth=256, mheight=256):
    image = Image.open(filename)
    w,h = image.size
    if w<=mwidth and h<=mheight:
        print(filename,'is OK.')
        return
    if (1.0*w/mwidth) > (1.0*h/mheight):
        scale = 1.0*w/mwidth
        new_im = image.resize((int(w/scale), int(h/scale)), Image.ANTIALIAS)

    else:
        scale = 1.0*h/mheight
        new_im = image.resize((int(w/scale),int(h/scale)), Image.ANTIALIAS)
    new_im.save(filename)
    new_im.close()

#opt = opts().parse()

imageName='./images/test3.jpg'

#process_image(imageName)

model = torch.load('../model/Stage3/model_10.pth',map_location=lambda storage,loc: storage)
img = cv2.imread(imageName)
print(type(np.array(img)))
input = torch.from_numpy(img.transpose(2, 0, 1)).float() / 256.
input = input.view(1, input.size(0), input.size(1), input.size(2))
input_var = torch.autograd.Variable(input).float()
output = model(input_var)
pred = getPreds((output[-2].data).cpu().numpy())[0] * 4
reg = (output[-1].data).cpu().numpy().reshape(pred.shape[0], 1)
print(pred,(reg + 1) / 2. * 256)
debugger = Debugger()
debugger.addImg((input[0].numpy().transpose(1, 2, 0)*256).astype(np.uint8))
debugger.addPoint2D(pred, (255, 0, 0))
debugger.addPoint3D(np.concatenate([pred, (reg + 1) / 2. * 256], axis = 1))
debugger.showImg(pause = True)
debugger.show3D()



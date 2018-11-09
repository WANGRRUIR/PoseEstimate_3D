nJoints = 16
accIdxs = [0, 1, 2, 3, 4, 5, 10, 11, 14, 15]
shuffleRef = [[0, 5], [1, 4], [2, 3],
              [10, 15], [11, 14], [12, 13]]
edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5],
         [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15],
         [6, 8], [8, 9]]

h36mImgSize = 224

outputRes = 64
inputRes = 256

eps = 1e-6

momentum = 0.0
weightDecay = 0.0
alpha = 0.99
epsilon = 1e-8


scale = 0.25
rotate = 30
hmGauss = 1
hmGaussInp = 20
shiftPX = 50
disturb = 10

dataDir = '../data'
mpii_img_dir='/home/wangrui/data/MPII/images'
h36m_img_dir='/home/wangrui/data/Human3.6M/images'
expDir = '../model'

nThreads = 4


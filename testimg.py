import imageio
import numpy as np
import glob
import scipy.ndimage as ndimage
from matplotlib import pyplot as plt

#
imgpaths  = []
for path in glob.glob('../data/stemcells/01/*.tif'):
    imgpaths.append(path)

imgpaths.sort()
print(imgpaths)
imgs = [ndimage.gaussian_filter(imageio.imread(path), 2) for path in imgpaths]

imgseries = np.stack(imgs)

print(imgseries.shape)
binarized = (imgseries > 1)
plt.imshow(binarized[0])
plt.show()
structure1 = np.ones((3, 13, 13))
structure2 = np.ones((5, 11, 11))
print('opening')
opened = ndimage.morphology.binary_opening(binarized, structure=structure2)
print('closing')
closed = ndimage.morphology.binary_closing(opened, structure=structure1)
print('closed')
for i in range(len(imgseries)):
    imageio.imwrite('data/stemcells/closed01/'+f'{i:03}'+'.png',closed[i]*255)
    

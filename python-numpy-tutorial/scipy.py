#Numpy提供了高性能的多维数组(矩阵)，以及计算和操作数组(矩阵)的基本工具。
#SciPy基于Numpy，提供了大量的计算和操作数组(矩阵)的函数，这些函数对于不同类型的科学和工程计算非常有用。
from scipy.misc import imread, imsave, imresize #misc这个库就是Miscellaneous杂项的缩写，有很多杂类的实用函数。
img = imread('assets/cat.jpg') #scipy会把image读入为一个numpy的array矩阵
print(img.dtype, img.shape) #uint8 (400, 248, 3)
img_tinted = img * [1, 0.95, 0.9] #x(1, 3)，在行方向上broadcast，意味着RGB中红色通道不变，绿色和蓝色通道像素值变小
img_tinted = imresize(img_tinted, (300, 300)) #前两维度变成(300, 300)，注意是imresize而不是reshape
imsave('assets/cat_tinted.jpg', img_tinted)

#scipy.io.loadmat和scipy.io.savemat可以读写mat文件

import numpy as np
from scipy.spatial.distance import pdist, squareform #space->spatial空间的，看库的名字也能看出来是计算空间中距离的
x = np.array([[0, 1], [1, 0], [2, 0]]) #每行都是个二维的坐标
print(x)
d = squareform(pdist(x, 'euclidean')) #可以简单理解为pdist求距离，squareform转换成距离矩阵，都是很强大的函数，之后细说
print(d) #scipy.spatial.distance.cdist计算的是不同集合中的点的距离

#关于用到的几个函数，之后在写代码的时候用到了会再做用法的学习和整理
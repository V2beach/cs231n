#Matplotlib是一个作图库，matplotlib.pyplot模块功能和MATLAB的作图功能类似。
import numpy as np
import matplotlib.pyplot as plt
print("---plotting---")
#Compute the x and y coordinates for points on a sine curve
#用一个sin曲线计算坐标值x和y
x = np.arange(0, 3 * np.pi, 0.1) #arange跟range差不多个事，都是范围的意思
y = np.sin(x)
plt.plot(x, y)
plt.show() #You must call plt.show() to make graphics appear.
print("---multiple lines---")
y_sin = np.sin(x)
y_cos = np.cos(x)
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sin and Cos')
plt.legend(['Sin', 'Cos']) #这个写法要学，string列表对应两个曲线做标注
plt.show()
print("---subplots---")
plt.subplot(2, 1, 1) #2行1列的第一个，rows/cols/index
plt.plot(x, y_sin)
plt.title('Sin')
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cos')
plt.show()
print("---imshow---")
from scipy.misc import imread, imresize
img = imread('assets/cat.jpg')
img_tinted = img * [1, 0.9, 0.3]
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.subplot(1, 3, 2)
plt.imshow(np.uint8(img_tinted)) #如果不是uint8，imshow会显示出问题，所以要用numpy提前转换
plt.subplot(1, 3, 3)
plt.imshow(img_tinted) #可以看这里
plt.show()
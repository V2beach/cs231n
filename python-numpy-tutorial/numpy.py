import numpy as np
#Numpy是Python中用于科学计算的核心库。
#它提供了高性能的多维数组对象，以及这些数组(矩阵)的相关工具。
print("---array---") #下面这里作为参考的中文笔记翻译错了。
a = np.array([1, 2, 3]) #index是非负整型数，维度数称为数组的阶rank，shape是元组。
print(type(a))
print(a.shape) #(3,)
print(a[0], a[1], a[2])
a[0] = 5
print(a)
#二维的
b = np.array([[1, 2, 3], [4, 5, 6]]) #发现我们可以用列表初始化array矩阵(*1)
print(b.shape)
print(b[0, 0], b[0, 1], b[1, 0])
print("---array functions---")
a = np.zeros((2, 2)) #输入的参数是一个元组，元组表示维度shape上面说过了(*2)
print(a)
b = np.ones((1, 2)) #全是1
print(b)
c = np.full((2, 2), 7)
print(c)
d = np.eye(2) #单位矩阵
print(d)
e = np.random.random((2, 2)) #np.random是个库
print("---array indexing---") #多种访问/索引数组的方法
print("---array indexing slicing---")
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]]) #从现在起初始化数组时候,后不写空格了，实在是累
b = a[:2, 1:3] #都是先行后列，初始化时最里面是行，slicing时前面是行
print(b, type(b))
print(a[0, 1]) #注意！slicing的新array是原来的一个视图，修改还是会改原数组(*3)
b[0, 0] = 77
print(a[0, 1])
#下面的内容是以前没接触过的，整型和切片方法访问数组的区别，
#这种方法要产生一个新的数组
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
row_r1 = a[1, :]
row_r2 = a[1:2, :]
print(row_r1, row_r1.shape)
print(row_r2, row_r2.shape)
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)
print(col_r2, col_r2.shape)
col_r1[0] = 0
print(a[0, 0])
#这里就会发现，无论是a[1, :]还是a[:, 1]都是产生一个shape(n,)的一维数组，低阶新数组！(*4)
#纯整型的就跟印象中的一样了，切片得到的总是原数组的一个子集，是原数组的view，整型得到的是一个利用数据产生的新数组。
a = np.array([[1,2], [3,4], [5,6]])
print(a, a.shape)
print(a[[0, 1, 2], [0, 1, 0]]) #输出的是a[0, 0], a[1, 1], a[2, 0]
print(a[[0, 0], [1, 1]]) #输出的是a[0, 1], a[0, 1]
print((a[0, 1], a[0, 1]))
#前面是行索引，后面是列索引，神奇的索引方式。(*5)
#下面是一个整型数组访问array比较好用的技巧trick。
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
print(a)
b = np.array([0, 2, 0, 1]) #将作为列索引
print(a[np.arange(4), b]) #分别对应0, 1, 2, 3行的第0, 2, 0, 1个元素
a[np.arange(4), b] += 10 #这个技巧感觉还是有用的
print(a)
print("---boolean array indexing---") #可以用布尔表达式作为索引下标，筛选出满足某种条件的所有元素，放入一个新的一维数组(*6)
a = np.array([[1,2], [3, 4], [5, 6]])
bool_idx = (a > 2) #会遍历原数组创建一个boolean数组作为下标/索引
print(bool_idx)
print(a[bool_idx]) #会构建一个新的一维数组[3, 4, 5, 6]存放所有满足条件的元素
print(a[a > 2])
print("---array datatypes---") #numpy的array会试图猜测数组的数据类型，但也可以自己指定，会猜测！(*7)
x = np.array([1, 2])
print(x.dtype) #int32
x = np.array([1.0, 2.0])
print(x.dtype)
x = np.array([1, 2], dtype=np.int64)
print(x.dtype)
#下面的是比较关键的内容，虽然上面的也很重要
print("---array math计算相关---")
#基本数学计算函数会对数组逐元素elementwise进行计算，既可以利用numpy重载的操作符来算，也可以使用np内置的函数：
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)
print(x + y) #对应元素相加
print(np.add(x, y))
print(x - y)
print(np.subtract(x, y))
print(x * y)
print(np.multiply(x, y))
print(x / y)
print(np.divide(x, y))
print(np.sqrt(x))
print("---array multiplication---")
#这里...我好像也理解错了，向量内积和矩阵乘法好像是十分相关的两个概念，就通过知乎上的，
#矩阵乘法的本质和矩阵求导术之类的东西来复习矩阵，多算多练，线性代数的东西都没用，现在学的矩阵分析的东西才是以后用得着的。
#这部分很多教程自己都写错了，随便搜两个内积外积关键词网上全是错误的解释，正确的理解是这样的：(*8)
#内积是指数量积，点乘！一般是指向量，对应求积结果求和，返回一个标量，
#外积是指张量积，在python里numpy的函数反而写作dot，但不能理解为点乘，这个dot就是矩阵乘法，
#但如果处理的是向量的话，dot就又是点乘，即内积，这个方法既是实例里的方法v.dot()，又是np里的方法np.dot()，
#跟内积有些相似的是对应元素相乘，结果还是同维度矩阵，python里的*和np.multiply都是这种运算。

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])
v = np.array([9,10])
w = np.array([11,12])
print(v.dot(w))
print(np.dot(v, w))
#Matrix dot multiply vector, rank 1 array
print(x.dot(v))
print(np.dot(x, v))
#Matrix dot multiply matrix, rank n array
print(x.dot(y))
print(np.dot(x, y))
print("---numpy array sum---")
x = np.array([[1,2],[3,4]])
print(np.sum(x))
print(np.sum(x, axis=0)) #axis坐标轴，分别对应按列和按行加和(*9)
print(np.sum(x, axis=1))
print(x)
print(x.T)
v = np.array([1,2,3])
print(v)
print(v.T) #Taking the transpose of a rank 1 array does nothing
print("---Broadcasting广播---") #让Numpy可以将不同大小的矩阵放在一起进行数学计算
#比如把向量加到矩阵的每一行
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x) #创建一个跟x的shape相同的空矩阵
#我们用循环来做
for i in range(4):
    y[i, :] = x[i, :] + v
print(y)
#当然是可以的，这样本质就是两个同纬度向量相加，但当矩阵很大的时候，循环效率会很低，换一种思路：
vv = np.tile(v, (4, 1)) #四个v（都是行向量，需要表示列向量要T转置，或者reshape）合在一起，这样也是个不错的办法
print(vv)
y = x + vv
print(y)
#但现在想避免生成一个vv，直接运算：
y = x + v #python的broadcasting机制允许直接加，可以把v加到每一行x上
print(y)
#下面总结一下使用broadcasting广播机制的规则：(*10)
#1.如果矩阵的阶数（这里可以翻译成秩吗），用1将小矩阵扩展到跟大矩阵相同尺寸；
#2.如果两个矩阵在某个维度大小一样，或者其中一个矩阵在某个维度上大小是1，那么就说这两个矩阵在这个维度上是相容的；
#3.如果两个矩阵在所有维度上都是相容的，他们才可以使用广播；
#4.如果两个矩阵的尺寸不同，那关注其中较大的那个尺寸，广播之后两个矩阵的尺寸和较大的那个一样（见上例）；
#5.在任何一个维度上，如果一个数组的大小是1，另一个数组的大小大于1，那么在这个维度上，第二个数组的运算对象就是对第一个数组进行的几次复制。
#支持广播机制的函数是全局函数，文档里有更详细的介绍，读了一下文档，下面另外有一些广播机制的例子。
v = np.array([1,2,3]) #(3,)其实就是(1, 3)，打印有区别，前者是[1,2,3]，后者是[[1,2,3]]
#print(np.reshape(v, (1, 3))) print(v)
w = np.array([4,5])
print(np.reshape(v, (3, 1)) * w) #如果需要把一个行向量转为列向量，不能用T，只能用reshape，这里的乘法其实是(3,1) * (1,2)，就是个矩阵外积乘法outer product，跟broadcast无关
x = np.array([[1,2,3], [4,5,6]]) #下面这个例子则用到了broadcast
print(x + v) #(2,3) + (1,3)，有个维度是一样的，所以x就相当于在两行上都加上v
print((x.T + w).T) #(3,2) + (1,2)，所以其实是在x.T每行加上w，最后转置，或者就是x + w.T，每列加w.T，一个意思
print(x + np.reshape(w, (2, 1))) #就是上面说的x + w.T
print(x * 2) #其实也是broadcast的思想，把2看成(1,1)的矩阵，就是在每行每列都乘2了
#Broadcasting typically makes your code more concise and faster, so you should strive to use it where possible.
#广播机制能够让你的代码更简洁更迅速，能够用的时候请尽量使用！
#很重要的一个技巧，以前没有注意过。

#自行尝试一下3x3的矩阵+2x3的矩阵
matrix_33 = np.array([[3, 3, 3], [3, 3, 3], [3, 3, 3]])
matrix_23 = np.array([[2, 2, 2], [2, 2, 2]]).T
print(matrix_23 + matrix_33) #这样不行的，只能是一维的...

# 广播为什么会让计算更迅速呢，诶？测试结果是用了one_loop反而相比two_loops变得更慢了（这个在速度比较那个NOTE里面说了）。(*10???√√√)
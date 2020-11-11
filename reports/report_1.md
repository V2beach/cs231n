# Lecture1 Course Introduction
课程相关的[Slides, Notes, Papers](http://cs231n.stanford.edu/syllabus.html)及[花书](https://mitpress.mit.edu/books/deep-learning)作为主要参考材料。

Lecture1根据[Course Materials](https://cs231n.github.io/python-numpy-tutorial/)完成了[Code of Python Numpy Tutorial](https://github.com/V2beach/cs231n/tree/main/Python_Numpy_Tutorial)。

会将我学习到的，课上讲到的核心内容、部分公式的推导、Assignments的代码原理和实现过程整理到本篇及后续的学习报告，以防止走马观花、边学边忘。

cs231n的课程名是Convolutional Neural Networks for Visual Recognition，即用于视觉识别的卷积神经网络，Lecture 1\~4通过CV和ML基本概念及算法原理讲解了Visual Recognition问题的处理流程和学习DL必要的前置ML知识，我也趁这段时间，借助花书和统计学习方法恶补了必要的数学知识；Lecture 5\~end才从CNN讲起，进入课程主题。

# Lecture2 Image Classification

### 图像识别的困难
对于人来说，“识别”功能简单至极，但从计算机视觉算法的角度来看，一张图片是以三维数组(e.g. 32x32x3)的形式被存储的，**矩阵中的元素Image[i, j, :]是该点(i, j)像素RGB三个通道的亮度值**，图像识别的困难如下，这些也是做CV任务时需要考虑的关键问题：
视角变化（Viewpoint variation）：同一个物体，摄像机可以从多个角度来展现。
大小变化（Scale variation）：物体可视的大小通常是会变化的。
形变（Deformation）：很多东西的形状并非一成不变，会有很大变化。
遮挡（Occlusion）：目标物体可能被挡住。
光照条件（Illumination conditions）：在像素层面上，光照的影响非常大。
背景干扰（Background clutter）：物体可能混入背景之中，使之难以被辨认。
类内差异（Intra-class variation）：一类物体的个体之间的外形差异很大，比如椅子。

### 向量范数度量图片差异
**范数，是具有“长度”概念的函数，为向量空间内的所有向量赋予大小**，L1和L2距离指的就是两个向量的差向量的L1和L2范数，其中向量的L2范数类似于矩阵的F范数。

<div align=center>
<img src="assets/L1L2_distance.png" width="70%" height="70%">
</div>


### Assignment1 k-NN

### 验证集用于超参数调优及交叉验证

# Lecture3 Loss Functions and Optimization

### 线性分类器

### 损失函数
分别是什么？模型，策略，算法，不用非得说三要素，只是要有全局的概念。优化这部分笔记要提上来，我觉得这样布置更合理。

### 最优化原理、梯度计算、梯度下降

### Assignment1 SVM

### Assignment1 Softmax

### SVM和Softmax比较及借助linear classifier demo整体理解

### 

# Lecture4 Neural Networks and Backpropagation

### 反向传播及相关知识

### Assignment1 Two-Layer Neural Network

（最后用typora生成目录
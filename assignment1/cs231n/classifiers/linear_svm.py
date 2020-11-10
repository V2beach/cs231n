from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1] # (n, m) matrix
    num_train = X.shape[0] # (n, 1) vectors
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1（Δ is 1
            if margin > 0: # 这个其实就是max(0, s_j - s_correct + 1)，写成分类讨论的形式更好理解
                loss += margin
                # 推了几遍公式之后终于发现，这里要减c - 1次xi，之后会把这部分公式也整理下来，根本搜不到一个靠谱的
                dW[:, y[i]] += -X[i, :] # 就这里，还是没搞明白这是咋求导的（或者应该叫微分？
                dW[:, j] += X[i, :] # 不用加.T转置吗？我晕了，后面重新看，看下面shape应该是不用加的啊，所以好多个笔记都写错了，GitHub那个是对的
                # print(dW.shape, X.shape) #(3073, 10) (500, 3073)维度是正确对应的
                # print(dW[:, j].shape, X[i, :].shape) # 都是(3073, )

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W) # 这里乘0.5是为了下面求导/微分后没有系数，正则化项，矩阵X的F范数||X||F对X求导结果是2X，结论先记一下，实在不知道咋推
    dW += reg * W # 上面不乘0.5，这里乘2也可以

    ### 照着笔记写完后想了想还是不行，
    ### svm loss包括之后梯度的计算是必须要理解的，knn的vector做法我也没完全搞懂，所以写到这里的时候暂时放了一下，去复习了“矩阵乘法的本质”并学习了“矩阵求导术/matrix cookbook”之类的东西，
    ### 另外我觉得今后要改变学习策略，
    ### 看完一个lecture就要做一个assignment的task，不然期间相隔太久，会导致——
    ### 看lecture的时候记不到心里去且不能真正理解，越看越飘，越看不懂，效率变很低；写assignment的时候因为基础不好，战线拉太长，又没了学习的兴趣，变得迷茫，需要有视频来作为强心剂。

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # TODO说的是，要计算梯度并存入dW，先计算损失再计算导数或许比同时计算麻烦（确实如此，数值方式最好同时计算），因此需要修改上面的代码，
    # 另外，矩阵求导算梯度公式的这块我还没太搞懂...只能用微分梯度分析法，不让用数值方法h=0.0001，在跟着下面第三篇笔记推导公式，
    # 主要在看的是官方笔记和https://zhuanlan.zhihu.com/p/28247951 https://zhuanlan.zhihu.com/p/21478575 https://doraemonzzz.com/2019/03/02/CS231%20作业1/ 这几篇。
    # https://blog.csdn.net/jichihui7464/article/details/80550616 这篇包括里面的两个参考链接，看这些就足够真正学会推导了，要想深入好像得学凸优化。
    # https://blog.csdn.net/lanchunhui/article/details/70991228 https://blog.csdn.net/silent_crown/article/details/78109461 看完也还是没理解为什么求导完就剩指示函数和x了。先写吧。

    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 写到这里我发现，我看的那篇WILL的知乎笔记完全是抄的lightaime的GitHub题解，所以我干脆直接去看原版了，笔记只是作为大纲

    num_classes = W.shape[1] # c
    num_train = X.shape[0] # m
    scores = X.dot(W) # X∈R^(m*n) W∈R^(n*c) S is XW∈R^(m*c) 即原本n个特征的m个样本分别有c个分类的得分s
    correct_class_scores = scores[range(num_train), list(y)].reshape(-1, 1) # 还是得画出来才能看明白怎么broadcast的
    margins = np.maximum(0, scores - correct_class_scores + 1) # 还是得画出来才能看明白怎么broadcast的
    margins[range(num_train), list(y)] = 0  # 还是得画出来才能看明白怎么broadcast的
    loss = np.sum(margins) / num_train + 0.5 * reg * np.sum(W * W) # reg是正则的那个系数

    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 历时三天我终于要写完了，公式也都会整理一遍，包括loops和vectorized的，lightaime这个向量化的操作挺牛逼的，不知道怎么想到的，系数矩阵实在是强，得再整理一下他的思路

    coeff_matrix = np.zeros((num_train, num_classes))
    coeff_matrix[margins > 0] = 1 # knn我做了实验，array > 0这种写法会返回一个同样维度的bool array，python牛逼啊，这个写法学习一下
    coeff_matrix[range(num_train), list(y)] = 0
    coeff_matrix[range(num_train), list(y)] = -np.sum(coeff_matrix, axis=1) # axis不加也是一样的，这里可以对axis加深理解，axis=1也就是shape[1]，对列进行加和就是对同一行上的每一列进行加和，这样来理解
    # print(coeff_matrix) # 一目了然
    dW = (X.T).dot(coeff_matrix) # 这里巧妙的做法会整理，另外这里只是算梯度dw，在这个函数外面再用SGD更新，即W - dw(gradient)
    dW = dW / num_train + reg * W

    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

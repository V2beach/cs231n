from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange
# import math
# from scipy.linalg import expm, logm # linear algebra 线性代数

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W) # (3073, 10)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 完整地完全独立完成独立调试跑通了代码，最后再跟之前几年的笔记的各种写法对照
    num_train = X.shape[0]
    num_classes = W.shape[1]
    # print(num_train, num_classes) # 维度值获取正确

    # 这一段留着待会向量化的时候再用，自己做！完全正确！
    # score = X.dot(W)
    # exp_sum = np.sum(np.exp(score), axis=1)
    # loss = np.sum(np.log(exp_sum) - score[range(0, num_train), list(y)]) # 找出所有yi这个操作我学会了，但是这里不能化简来做，求梯度会用到未化简的中间值P
    # print(score.shape, exp_sum.shape, loss.shape, type(loss))
    for i in range(num_train): # 所有样本的Li都要计算进来，合称为Loss，之后再/num + reg，先把loss的写出来，再加进去gradient的计算
        score = X[i, :].dot(W) # (500, 10)
        exp_sum = np.sum(np.exp(score), axis=0) # (500, )
        probability = np.exp(score[0: num_classes]) / exp_sum # 第i个样本每个类别的对应概率
        loss += -np.log(probability[y[i]])
        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] += (probability[j] - 1) * X[i, :]
            else:
                dW[:, j] += probability[j] * X[i, :]
    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W * W) # F范数求导
    dW += 2 * reg * W # 得到2W

    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]

    score = X.dot(W) # (500, 10)
    exp_sum = np.sum(np.exp(score), axis=1).reshape((-1, 1)) # (500, )，这里必须要转成列向量，经历了这么多天，自己完整实现基础的机器学习算法感觉太爽了
    probability = np.exp(score) / exp_sum # (500, 10) / (500, 1) = (500, 10)
    loss = np.sum(-np.log(probability[range(0, num_train), list(y)]) / num_train + reg * np.sum(W * W)) # 就是糅合在一起了
    probability[range(0, num_train), list(y)] -= 1 # 推导最长的那步偏导，作为中间值
    dW = (X.T).dot(probability) / num_train + 2 * reg * W

    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

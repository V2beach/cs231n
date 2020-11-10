from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        # kNN的O(1)训练过程只是记住了所有的训练数据
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                # 这个assignment的形式实在是太棒了，不仅有助于理清代码思路，避免不必要的工作量，让python/nndl新人陷入混乱，
                # 还可以为今后学生自己书写的代码提供一个非常漂亮的模板，简直完美~
                # 每两个图片的维度都是相同的，所以直接相减，求平方和，开方就行，想一下求二维点的距离也确实是这么操作的，最后开方是因为z^2 = x^2 + y^2
                
                dists[i, j] = np.sqrt(np.sum(np.square(self.X_train[j] - X[i])))

                # pass

                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            
            # print((self.X_train - X[i]).shape) # (5000, 3072) dists一行是test中一个点相对于train所有点的距离，所以该是(1, 5000)的向量，那么下式是怎么转换成行向量的呢？
            dists[i] = np.sqrt(np.sum(np.square(self.X_train - X[i]), axis=1)) # axis=0是列相加，axis=1是行相加，同一个图片的所有像素是1x3072，在同一行
            # print(dists[i].shape) # (5000, )，所以其实是np.sum加和后向量变成一维，而numpy里默认是行向量，如果需要列向量需要reshape手动转，不能.T
            
            # pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 想了半天，脑袋僵住了，之后恍然大悟，矩阵的很多运算思路跟实数运算是一样的！这里就是差平方公式(a - b)^2 = a^2 - 2ab + b^2，太久不用了
        dists = np.sqrt(np.sum(np.square(X), axis=1).reshape(-1, 1) - 2 * X.dot(self.X_train.T) + np.sum(np.square(self.X_train), axis=1)) #这种方法巨快无比
        # 我把这个公式的解释写一下

        # pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # 先找到k个最近的相邻点
            # 关于提到的numpy.argsort函数，https://www.jianshu.com/p/64c607d49528 这篇博客做了个详尽的介绍，有C++Algorithm里sort的感觉，
            close_y_order = np.argsort(dists[i], axis=0, kind='quicksort') # 只有一维，所以axis就是0，如果是二维想排列同行的axis=1，三维axis需为2，注意argsort函数返回的是从小到大排列过后列表元素的索引，sort返回的才是元素本身
            closest_y = self.y_train[close_y_order[0:k]] # 从索引转为labels，list是可以用一个列表来索引值的

            # pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # 然后找出k个里面出现次数最高的一个类别，即投票
            # 关于bincount函数，这个作者讲得非常详细了https://blog.csdn.net/xlinsist/article/details/51346523 ，整理report的时候结合自己写的代码把这个再捋一遍，用熟
            # 而argmax返回的恰好是第一个最大值的下标！恰好是第一个，也就是较小的
            size = np.max(closest_y) + 1 # 因为是从0开始计数的，要统计每个数的出现次数，实现方式是以这个数为下标，bins中存次数，所以size = max + 1（bins垃圾箱
            bins = np.zeros(size)
            for j in range(0, k):
                bins[closest_y[j]] += 1
            vote_y_order = np.argsort(bins, axis=0, kind='quicksort')[::-1] # 存入的就是类别，这里写错了，如果需要从大到小排序
            y_i_pred = vote_y_order[0]
            for j in range(0, size):
                if bins[y_i_pred] == bins[vote_y_order[j]]:
                    y_i_pred = min(y_i_pred, vote_y_order[j]) # 选一个类别idx更小的
            y_pred[i] = y_i_pred

            y_pred[i] = np.argmax(np.bincount(closest_y)) # 上面那一片实现的就是这个函数的一小部分，两种结果相同，这种写法更好，因为numpy内部应该会有优化
            # print(vote_y_order, y_i_pred, " ", y_pred[i]) # debug

            # pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred

from keras.datasets import mnist
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class test_knn_model(object):

    def __init__(self, model):
        self.model = model

    def small_test_on_mnist_and_comparing_with_sklearn(self):
        print("\n\nSmall test on mnist and comparing with sklearn\n")
        (train_x, train_y), (test_x, test_y) = mnist.load_data()
        train_x = train_x[:700]
        train_y = train_y[:700]
        test_x = test_x[200:500]
        test_y = test_y[200:500]
        train_x = train_x.reshape(len(train_x), 784)
        test_x = test_x.reshape(len(test_x), 784)
        print("train_x_shape", train_x.shape)
        print("test_x_shape", test_x.shape)

        MyKnn = self.model
        print("\nBuilded model: n_neighbors = {}, weights = {}, leaf_size = {}".format(MyKnn.n_neighbors, MyKnn.weights, MyKnn.leaf_size))
        print("Fitting model")
        MyKnn.fit(train_x, train_y)

        print("\nCreating sklearn with same params, fitting it")
        SklearnKnn = KNeighborsClassifier(algorithm = 'ball_tree', n_neighbors = MyKnn.n_neighbors, weights = MyKnn.weights, leaf_size = MyKnn.leaf_size)
        SklearnKnn.fit(train_x, train_y)

        print("\nJust for check give for predict train_x and accuracy should be 1.0")
        acc = accuracy_score(MyKnn.predict(train_x), train_y)
        if acc != 1.0:
            print("Accuracy != 1.0")
        else:
            print("Accuracy = 1.0, all is ok")

        print("\nMake predict and compare 2 arrays")
        a = MyKnn.predict(test_x)
        b = SklearnKnn.predict(test_x)
        if np.all(a == b):
            print("No error in predict")
        else:
            print("Error in predict")

        print("\nMake predict_proba and compare 2 matrices, using np.around(..., 5)")
        a = np.around(MyKnn.predict_proba(test_x), 5)
        b = np.around(SklearnKnn.predict_proba(test_x), 5)
        if np.all(a == b):
            print("No error in predict_proba")
        else:
            print("Error in predict_proba")

        print("\nMake kneighbors and compare, n_neighbors = 5")
        a, b = MyKnn.kneighbors(test_x[:10], n_neighbors = 5)
        c, d = SklearnKnn.kneighbors(test_x[:10], n_neighbors = 5)
        a = np.around(a, 5)
        c = np.around(c, 5)
        if np.all(a == c) and np.all(b == d):
            print("No error in kneighbors")
        else:
            print("Error in kneighbors")

        MyKnn.fitted = 0

    def test_for_exeptions(self):
        print("\n\nTest on exeptions\n")
        MyKnn = self.model
        print("\nBuilded model: n_neighbors = {}, weights = {}, leaf_size = {}".format(MyKnn.n_neighbors, MyKnn.weights, MyKnn.leaf_size))

        print("\nTry to give as X string for fit:")
        MyKnn.fit("just_for_test", ['A'])

        print("\nTry predict, without fitting:")
        MyKnn.predict(QueryMatrix = [[0]])
        print("\nTry predict_proba, without fitting:")
        MyKnn.predict_proba(QueryMatrix = [[0]])
        print("\nTry kneighbors, without fitting:")
        MyKnn.kneighbors(QueryMatrix = [[0]], n_neighbors = 1)

        print("\nGen random dataset, X_small will be 2d point, with function np.random.rand. len(X_small) = n_neighbors (in builded model)")
        X_small = np.random.rand(MyKnn.n_neighbors, 2)
        Y_small = np.random.rand(MyKnn.n_neighbors)
        print("X_small.shape", X_small.shape)
        print("Y_small.shape", Y_small.shape)
        print("\nTry to fit with X = X_small, Y = np.random.rand(n_neighbors + 1) - len(X) != len(Y) should be:")
        MyKnn.fit(X_small, np.random.rand(MyKnn.n_neighbors + 1))
        print("\nFitting model with X_small, Y_small")
        MyKnn.fit(X_small, Y_small)

        print("\nTry predict with QueryMatrix as integer:")
        MyKnn.predict(QueryMatrix = 5)

        print("\nTry predict_proba with QueryMatrix as string:")
        MyKnn.predict(QueryMatrix = "just_for_test")
        print("\nDifferent exeptions, because integer transforms into np array, but then shapes are not equal")

        print("\nTry predict_proba with QueryMatrix = np.random.rand(100, 3), wrong shapes:")
        MyKnn.kneighbors(QueryMatrix = np.random.rand(100, 3), n_neighbors = 1)

        print("\nTry kneighbors with QueryMatrix = np.random.rand(100, 2), n_neighbors = len(X_small) + 1:")
        MyKnn.kneighbors(QueryMatrix = np.random.rand(100, 2), n_neighbors = len(X_small) + 1)

        MyKnn.fitted = 0

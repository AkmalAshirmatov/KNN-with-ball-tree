# idea: https://en.wikipedia.org/wiki/Ball_tree
# choosing order of visiting children, while searching k-neighbors https://upcommons.upc.edu/bitstream/handle/2117/76382/hpgm_15_1.pdf?sequence=1&isAllowed=y

import numpy as np

class KnnBalltreeClassifier(object):

    class Node(object):

        def __init__(self, root, radius, left_child, right_child, points, is_leaf, id_max_spread):
            self.root = root
            self.radius = radius
            self.left_child = left_child
            self.right_child = right_child
            self.points = points
            self.is_leaf = is_leaf
            self.id_max_spread = id_max_spread

    def ball_tree(self, points, max_leaf_size, X):
        if len(points) == 0:
            return None

        if len(points) <= max_leaf_size:
            return self.Node(points[0], None, None, None, points, 1, 0)

        nX = X[np.array(points)]
        id_max_spread = np.argmax(np.ndarray.max(nX, axis = 0) - np.ndarray.min(nX, axis = 0))
        points.sort(key = lambda elem : X[elem][id_max_spread])
        root_id = len(points) // 2
        root = points[root_id]
        radius = max([self.L2(X[id], X[root]) for id in points])

        left_child = self.ball_tree(points[:root_id], max_leaf_size, X)
        right_child = self.ball_tree(points[root_id + 1 :], max_leaf_size, X)

        return self.Node(root, radius, left_child, right_child, [], 0, id_max_spread)

    def L2(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def update(self, my_point, new_point, my_neighbors, n_neighbors, X):
        my_neighbors.append(new_point)
        my_neighbors.sort(key = lambda elem : self.L2(X[elem], my_point))
        if len(my_neighbors) > n_neighbors:
          my_neighbors.pop(-1)

    def ball_tree_search(self, node, my_point, my_neighbors, n_neighbors, X):
        if node.is_leaf == 1:
            for new_point in node.points:
              self.update(my_point, new_point, my_neighbors, n_neighbors, X)
            return

        if len(my_neighbors) > 0 and self.L2(my_point, X[node.root]) - node.radius >= self.L2(my_point, X[my_neighbors[-1]]):
            return

        self.update(my_point, node.root, my_neighbors, n_neighbors, X)

        other_child = None
        if my_point[node.id_max_spread] < X[node.root][node.id_max_spread]:
            self.ball_tree_search(node.left_child, my_point, my_neighbors, n_neighbors, X)
            other_child = node.right_child
        else:
            self.ball_tree_search(node.right_child, my_point, my_neighbors, n_neighbors, X)
            other_child = node.left_child

        if other_child != None and (len(my_neighbors) < n_neighbors or abs(my_point[node.id_max_spread] - X[node.root][node.id_max_spread]) < self.L2(my_point, X[my_neighbors[-1]])):
            self.ball_tree_search(other_child, my_point, my_neighbors, n_neighbors, X)


    def get_weight(self, p1, p2):
        if self.weights == 'uniform':
            return 1
        elif self.weights == 'distance':
            return 1 / (self.L2(p1, p2) + 1e-20)
        else:
            return self.weights(self.L2(p1, p2))

    def check_fitted(self):
        if self.fitted == 0:
            print("Model is not fitted yet")
            return False
        return True

    def check_X_size_and_n_neighbors(self, X, n_neighbors):
        if len(X) < n_neighbors:
            print("Expected n_neighbors <= n_samples,  but n_samples = {}, n_neighbors = {}".format(len(self.X), n_neighbors))
            return False
        return True

    def check_QueryMatrix(self, QueryMatrix):
        if self.X.ndim != QueryMatrix.ndim or self.X[0].shape != QueryMatrix[0].shape:
            print("Wrong shapes X and QueryMatrix")
            return False
        return True

    def check_size_X_Y(self, X, Y):
          if len(X) != len(Y):
              print("len(X) != len(Y), something wrong :)")
              return False
          return True

    def __init__(self, n_neighbors=1, weights='uniform', leaf_size=30):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.leaf_size = leaf_size
        self.root = None
        self.X = None
        self.Y = None
        self.fitted = 0

    def fit(self, X, Y):
        try:
            X = np.array(X, dtype = float)
        except ValueError:
            print("Wrong type of X")
            return
        if self.check_size_X_Y(X, Y) == False:
            return
        if self.check_X_size_and_n_neighbors(X, self.n_neighbors) == False:
            return
        self.root = self.ball_tree([i for i in range(len(X))], self.leaf_size, X)
        self.X = X.copy()
        self.Y = Y.copy()
        self.fitted = 1

    def predict(self, QueryMatrix):
        try:
            QueryMatrix = np.array(QueryMatrix, dtype = float)
        except ValueError:
            print("Wrong type of QueryMatrix")
            return
        if self.check_fitted() == False:
            return
        if self.check_QueryMatrix(QueryMatrix) == False:
            return
        ans = []
        for query_id in range(len(QueryMatrix)):
            query = QueryMatrix[query_id]
            my_neighbors = []
            self.ball_tree_search(self.root, query, my_neighbors, self.n_neighbors, self.X)
            weights_classes = dict().fromkeys(self.Y, 0)
            for id in my_neighbors:
                weights_classes[self.Y[id]] += self.get_weight(query, self.X[id])
            ans.append(max(weights_classes, key = weights_classes.get)) # https://stackoverflow.com/a/280156
        return np.array(ans)

    def predict_proba(self, QueryMatrix):
        try:
            QueryMatrix = np.array(QueryMatrix, dtype = float)
        except ValueError:
            print("Wrong type of QueryMatrix")
            return
        if self.check_fitted() == False:
            return
        if self.check_QueryMatrix(QueryMatrix) == False:
            return
        ans = []
        for query_id in range(len(QueryMatrix)):
            query = QueryMatrix[query_id]
            my_neighbors = []
            self.ball_tree_search(self.root, query, my_neighbors, self.n_neighbors, self.X)
            weights_classes = dict().fromkeys(self.Y, 0)
            for id in my_neighbors:
                weights_classes[self.Y[id]] += self.get_weight(query, self.X[id])
            sum_weights = 0
            for (key, weight) in weights_classes.items():
                sum_weights += weight
            ans.append(np.array([weight / sum_weights for (key, weight) in sorted(weights_classes.items())]))
        return np.array(ans)

    def kneighbors(self, QueryMatrix, n_neighbors):
        try:
            QueryMatrix = np.array(QueryMatrix, dtype = float)
        except ValueError:
            print("Wrong type of QueryMatrix")
            return
        if self.check_fitted() == False:
            return
        if self.check_X_size_and_n_neighbors(self.X, n_neighbors) == False:
            return
        if self.check_QueryMatrix(QueryMatrix) == False:
            return
        neigh_dist = []
        neigh_indarray = []
        for query_id in range(len(QueryMatrix)):
            query = QueryMatrix[query_id]
            my_neighbors = []
            self.ball_tree_search(self.root, query, my_neighbors, n_neighbors, self.X)
            my_neighbors_dist = [self.L2(self.X[id], query) for id in my_neighbors]
            neigh_dist.append(np.array(my_neighbors_dist))
            neigh_indarray.append(np.array(my_neighbors))
        return np.array(neigh_dist), np.array(neigh_indarray)

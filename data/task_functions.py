import numpy as np
from enum import Enum
from my_common.my_helper import is_2d_symmetric, is_ndarray
from util.constants import TaskType, GraphFunction


class GraphTasks:
    @classmethod
    def get_function(cls, graph_function):
        # Make sure it's a GraphFunction enum
        if isinstance(graph_function, str):
            graph_function = GraphFunction[graph_function]
        assert isinstance(graph_function, GraphFunction)

        if graph_function == GraphFunction.first_degree:
            return cls.first_degree
        if graph_function == GraphFunction.max_degree:
            return cls.max_degree
        if graph_function == GraphFunction.det_adj:
            return cls.det_adj

    @classmethod
    def get_task_info(cls, graph_function):
        """
        return: task type, target dimension
        """
        if isinstance(graph_function, str):
            graph_function = GraphFunction[graph_function]
        assert isinstance(graph_function, GraphFunction)

        if graph_function == GraphFunction.first_degree:
            return TaskType.regression, 1
        elif graph_function == GraphFunction.max_degree:
            return TaskType.regression, 1
        elif graph_function == GraphFunction.det_adj:
            return TaskType.regression, 1

    @staticmethod
    def first_degree(adjmat):
        assert is_ndarray(adjmat)
        assert is_2d_symmetric(adjmat)
        return np.sum(adjmat[0])

    @staticmethod
    def max_degree(adjmat):
        assert is_ndarray(adjmat)
        assert is_2d_symmetric(adjmat)

        degrees = np.sum(adjmat, axis=1)
        return np.amax(degrees)

    @staticmethod
    def det_adj(adjmat):
        assert is_ndarray(adjmat)
        assert adjmat.ndim == 2 and adjmat.shape[0] == adjmat.shape[1]
        return np.linalg.det(adjmat)


if __name__ == "__main__":
    # GraphTasks.get_task_info('foo')
    a = np.array([1., 7, -33, 17, 7, 5, -5, 4, -4]).reshape(3, 3)
    GraphTasks.det_adj(a)

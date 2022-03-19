# Barabasi-Albert Graphs

import numpy as np
import random
import inspect
from igraph import GraphBase  # Samples Barabasi-Albert
from util.constants import BaM, DEFAULT_RANDOM_GENERATION_SEED


class BaData:
    """
    Wraps around igraph's Barabasi generation function, conveniently specifying its parameters for our use cases
    See page 118 of https://igraph.org/python/doc/python-igraph.pdf
    """
    def __init__(self, graph_size, n_graphs, m_type, m_kwargs=None, ba_kwargs=None,
                 random_state=DEFAULT_RANDOM_GENERATION_SEED):
        """
        Constructor ensures proper types:
        :param graph_size: is integer > 2
        :param m_type: Enum BaM,
        :param m_kwargs: Dictionary or None
        :param ba_kwargs: Dictionary or None

        Details:
        graph_size is needed to create m_list properly
        **m_kwargs used in create_m_list() , options for creating an m-list.
          >> The m_list can approximate functionality of the fitness model and gives us flexibility
        **ba_kwargs used in Barabasi call.

        Programmer notes:
           - always cast arguments you want to use, b/c they  may come as strings from command line
           - defer assertions and validation to Barabasi function
        """
        self.graph_size = int(graph_size)
        self.n_graphs = int(n_graphs)
        assert self.graph_size > 2, "Expect BA graphs to have at LEAST three vertices..."
        assert self.n_graphs > 0
        assert isinstance(m_type, BaM)
        assert isinstance(m_kwargs, dict) or m_kwargs is None
        assert isinstance(ba_kwargs, dict) or ba_kwargs is None
        assert isinstance(random_state, int)
        self.random_state = random_state
        self.m_type = m_type

        # Convert kwargs vars to dictionaries
        if m_kwargs is None:
            self.m_kwargs = {}
        else:  # already a dictionary
            self.m_kwargs = m_kwargs
        if ba_kwargs is None:
            self.ba_kwargs = {}
        else:
            self.ba_kwargs = ba_kwargs

    def _constant_m_list(self, const=1):
        """
        Called when m_type is BaM.constant: cast const as int and return
        """
        m_val = int(const)
        assert m_val > 0, "Expected positive int"
        return m_val

    def _bernoulli_m_list(self, num_leading_const=10, const_val=1, prob=0.5):
        """
        Called when m_type is BaM.bernoulli
        Return [constant, ..., constant, samples from 1 + bernoulli]
        """
        # Cast (from command line)
        num_leading_const = int(num_leading_const)
        const_val = int(const_val)
        prob = float(prob)
        assert num_leading_const > 0 and const_val > 0
        assert 0 < prob < 1

        # Make sure num_leading_const is `small enough`
        if num_leading_const > self.graph_size:
            num_leading_const = self.graph_size

        if num_leading_const > self.graph_size/2.:
            print(f"Warning: large num_leading_const = {num_leading_const} for graph_size = {self.graph_size}")

        # [constant, ..., constant, samples from 1 + bernoulli]
        constants = np.repeat(const_val, num_leading_const)
        # bernoulli + 1 (we can't add 0 edges)
        bernoullis = 1 + np.random.binomial(1, prob, size=self.graph_size - num_leading_const)

        return np.concatenate([constants, bernoullis]).tolist()

    def create_m_list(self):
        """
        Return a scalar or list to be entered into the Barabasi function
        Wraps around worker functions
        """
        if self.m_type == BaM.constant:
            assert len(self.m_kwargs) <= 1, "Expected m_kwargs to be length 0 or 1 when BaM.constant"
            m_val = self._constant_m_list(**self.m_kwargs)
        elif self.m_type == BaM.bernoulli:
            m_val = self._bernoulli_m_list(**self.m_kwargs)
        else:
            raise NotImplementedError("unexpected m_type")

        if isinstance(m_val, list):
            if len(m_val) > 0:
                return m_val
            else:
                return m_val[0]
        elif isinstance(m_val, int):
            return m_val

    def sample_ba_graphs(self):
        """
        Return Barabasi-Albert graph
        """
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        m_list = self.create_m_list()
        out_list = list()
        for ii in range(self.n_graphs):
            graph = GraphBase.Barabasi(n=self.graph_size, m=m_list, **self.ba_kwargs)
            out_list.append(graph)

        return out_list

    @classmethod
    def helper(self):
        _str = "-"*55
        _str += "\nBarabasi Albert wrapper class, functions with signatures\n"
        _str += f"_constant_m_list: {inspect.signature(self._constant_m_list)} \n"
        _str += f"_bernoulli_m_list: {inspect.signature(self._bernoulli_m_list)} \n"
        _str += "-"*55
        return _str

    def __str__(self):
        _str = "Barabasi Albert Wrapper\n"
        _str += f"m_type = {self.m_type}\n"
        _str += f"graph_size = {self.graph_size}\n"

        return _str

    # ----------- graph ID
    @classmethod
    def ba_style_graph_id(self, graph_size, n_graphs, m_type, m_kwargs, ba_kwargs, random_state):
        """ Workhorse method to get the graph ID string with or without instantiating the object"""
        if isinstance(m_type, str):
            m_type = BaM[m_type]
        id_params = [n_graphs, graph_size, 'm',  m_type.name]

        for kk, vv in m_kwargs.items():
            id_params += [kk, vv]

        # Add BA kwargs info
        id_params += ['b']  # b denotes BA part of the string.
        for kk, vv in ba_kwargs.items():
            id_params += [kk, vv]

        id_params += ['seed', random_state]

        id_str = "_".join(map(str, id_params))
        return id_str


if __name__ == "__main__":
    # Uncomment to see signatures from command line
    # print(BaData.helper())

    # Testing
    import argparse
    from my_common.my_helper import args_list_to_dict

    parser = argparse.ArgumentParser()
    parser.add_argument("--ba_m_args", nargs="+",
                        help=f"Arguments for the m_list functions in BaData\n {BaData.helper()}")
    parser.add_argument("--ba_ba_args", nargs="+",
                        help=f"Additional KW Arguments for Barabasi Albert generation beyond n and m")
    args = parser.parse_args()

    m_kwargs = args_list_to_dict(args.ba_m_args)
    ba_kwargs = args_list_to_dict(args.ba_ba_args)
    bg = BaData(5, BaM.bernoulli, m_kwargs, ba_kwargs)
    bg.sample_ba_graphs()


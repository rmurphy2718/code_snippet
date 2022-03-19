# General Erdos-Renyi

import numpy as np
import random
import inspect
import networkx as nx
from util.constants import ErP, DEFAULT_RANDOM_GENERATION_SEED


class ErData:
    """
    Wraps around Erdos-Renyi graphs of networkx
    """
    def __init__(self, graph_size, n_graphs, p_type, p_kwargs=None,
                 random_state=DEFAULT_RANDOM_GENERATION_SEED):
        """
        Constructor ensures proper types:
        :param graph_size: is integer > 2
        :param p_type: Enum ErP,
        :param p_kwargs: Dictionary or None

        Details:
        **p_kwargs is used to determine how `p` gets decided from graph to graph

        Programmer notes:
           - always cast arguments you want to use, b/c they may come as strings from command line
           - defer assertions and validation to nx function
        """
        self.graph_size = int(graph_size)
        self.n_graphs = int(n_graphs)
        assert self.graph_size > 2, "Expect ER graphs to have at LEAST three vertices..."
        assert self.n_graphs > 0
        assert isinstance(p_type, ErP)
        assert isinstance(p_kwargs, dict) or p_kwargs is None
        assert isinstance(random_state, int)
        self.random_state = random_state
        self.p_type = p_type

        # Convert kwargs vars to dictionaries
        if p_kwargs is None:
            self.p_kwargs = {}
        else:  # already a dictionary
            self.p_kwargs = p_kwargs

    def _constant_p(self, const=0.5):
        p_val = float(const)
        assert 0 < p_val < 1, "Expected val in (0, 1)"
        return np.repeat(p_val, self.n_graphs).tolist()

    def _rand_p(self, low=0.2, high=0.85):
        """
        Sample random uniform between "low" and "high", return as list
        """
        assert 0 < low < high
        assert high <= 1
        samples = low + (high - low) * np.random.rand(self.n_graphs)
        return samples.tolist()

    def create_p_list(self):
        """
        Return a scalar or list to be entered into the Erdos-Renyi function
        Wraps around worker functions
        """
        if self.p_type == ErP.const:
            assert len(self.p_kwargs) <= 1, "Expected p_kwargs to be length 0 or 1 when ErP.constant"
            p_val = self._constant_p(**self.p_kwargs)
        elif self.p_type == ErP.rand:
            p_val = self._rand_p(**self.p_kwargs)
        else:
            raise NotImplementedError("unexpected p_type")

        if isinstance(p_val, list) and len(p_val) == self.n_graphs:
            return p_val
        else:
            raise RuntimeError(f"p_val should be a list with an element for each graph (should be { self.n_graphs})")

    def sample_graphs(self):
        """
        Return ER graphs
        """
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        p_list = self.create_p_list()
        out_list = list()
        for pp in p_list:
            graph = nx.gnp_random_graph(n=self.graph_size, p=pp)
            out_list.append(graph)

        return out_list

    @classmethod
    def helper(self):
        _str = "-"*55
        _str += "\nErdos-Renyi wrapper class, functions with signatures\n"
        _str += f"_constant_p_list: {inspect.signature(self._constant_p)} \n"
        _str += f"_rand_p_list: {inspect.signature(self._rand_p)} \n"
        _str += "-"*55
        return _str

    def __str__(self):
        _str = "Erdos-Renyi wrapper class\n"
        _str += f"p_type = {self.p_type}\n"
        _str += f"graph_size = {self.graph_size}\n"

        return _str

    # ----------- graph ID
    @classmethod
    def er_style_graph_id(cls, graph_size, n_graphs, p_type, p_kwargs, random_state):
        """ Workhorse method to get the graph ID string with or without instantiating the object"""
        if isinstance(p_type, str):
            p_type = ErP[p_type]
        id_params = [n_graphs, graph_size, 'p',  p_type.name]

        for kk, vv in p_kwargs.items():
            id_params += [kk, vv]

        id_params += ['seed', random_state]

        id_str = "_".join(map(str, id_params))
        return id_str


if __name__ == "__main__":

    # Testing
    from configs.graph_config import GraphDataArgParser

    args_str = "-data ger -gf det_adj --n_splits 2 --sparse "
    args_str += '-ger_N 20 -ger_n 10 --ger_p_type rand --ger_p_args high 0.9 low 0.5 '
    args_str += '--sample_random_state 3'

    args = GraphDataArgParser(args_str.split())

    erg = ErData(graph_size=args.ger_config['n_vert'],
           n_graphs=args.ger_config['n_graphs'],
           p_type=args.ger_config['p_type'],
           p_kwargs=args.ger_config['p_kwargs'],
           random_state=args.sample_random_state)

    print(erg)
    erg.sample_graphs()

    args.get_graph_id_string()
    args.get_data_path()



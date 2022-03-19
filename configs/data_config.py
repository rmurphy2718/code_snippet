from abc import abstractmethod
from argparse import ArgumentParser
from configs.config_argparser import ArgParser

from util.constants import DEFAULT_RANDOM_GENERATION_SEED

#TODO: Use 'append' to separate testing config from training config
class DataArgParser(ArgParser):

    @classmethod
    def add_common_args(cls, parser):
        assert isinstance(parser, ArgumentParser)

        parser.add_argument('-data', '--experiment_data', required=True, type=str, help='Data for experiment')
        parser.add_argument('--sample_random_state', default=DEFAULT_RANDOM_GENERATION_SEED, type=int,
                            help='random state for sampling graphs.')

    @abstractmethod
    def get_graph_id_string(self):
        """
        Get graph id string
        :return: The graph id string representing the configurations of graphs
        """
        pass

    @abstractmethod
    def get_data_path(self):
        pass

if __name__ == '__main__':
    args = DataArgParser(args=['hello'])  # Should get not instantiatiable error
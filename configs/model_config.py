from abc import abstractmethod
from argparse import ArgumentParser
from configs.config_argparser import ArgParser

class ModelArgParser(ArgParser):

    @classmethod
    def add_common_args(cls, parser):
        assert isinstance(parser, ArgumentParser)

    @abstractmethod
    def get_model_id_list(self):
        """
        Get model id list
        :return: The model id list representing the configurations of models
        """
        pass

if __name__ == '__main__':
    args = ModelArgParser(args=['hello'])  # Should get not instantiatiable error
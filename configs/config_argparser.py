import argparse
from abc import ABC, abstractmethod

class ArgParser(ABC):
    """
    Parent class that implements a general argument parser of data/model configurations.

    Specific argument parsers should inherit and implement "add_args" and "assign_parsed" functions,
    which will be used by the parent parser in Config.
    """
    def __init__(self, args=None, parsed_args=None):
        super().__init__()
        self.parser = argparse.ArgumentParser("Argument parser for data/model configurations")
        self.add_common_args(self.parser)
        self.add_args(self.parser)
        if parsed_args is None:
            parsed_args = self.parser.parse_args(args)
        # Assign parsed args to object
        try:
            self.assign_parsed(parsed_args)
        except AttributeError:
            print("-"*10)
            print("Did you forget a flag?\nDid you add new parser to MODEL_ARG_PARSERS list in config.py?")
            print("-" * 10)
        finally:
            pass  # Automatically prints usual warning message and quits if there's an error.

    @classmethod
    @abstractmethod
    def add_common_args(cls, parser):
        """
        Add common arguments shared among different kinds of data and models (e.g., n_splits, activations) to the parser
        separately from `add_args` to prevent collision of duplicate common arguments.

        :param parser: ArgumentParser
        """
        pass

    @classmethod
    @abstractmethod
    def add_args(self, parser):
        pass

    @abstractmethod
    def assign_parsed(self, parsed_args):
        pass


if __name__ == '__main__':
    args = ArgParser(args=['hello'])  # Should get not instantiatiable error
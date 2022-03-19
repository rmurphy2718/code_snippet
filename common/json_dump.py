import ast
import enum
import logging
import re
import time

import numpy as np
import pandas as pd
import torch

from common.helper import tdiv, print_stats


class JsonDump:
    def __init__(self, filename):
        print_stats("creating_json_dump", to=filename)
        self.logger = self.get_logger(filename)
        self.start_time = time.time()

    def get_logger(self, path):
        logger = logging.getLogger(path)
        logger.propagate = False

        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(path, mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    @staticmethod
    def _format(t):
        k, v = t
        if torch.is_tensor(v):
            v = float(v)
        if type(v) == tuple:
            v = tdiv(v)
        if isinstance(v, enum.Enum):
            v = str(v.name)
        if isinstance(v, np.generic):
            v = v.item()

        if isinstance(k, enum.Enum):
            k = str(k.name)

        return k, v

    def add(self, **info):
        self.logger.info(dict(map(JsonDump._format, info.items())))

    def no_format_add(self, **info):
        self.logger.info(info.items())

    def str_add(self, my_str):
        self.logger.info(my_str)

    @staticmethod
    def read(file, ret_df=True):
        with open(file, "r") as fp:
            lines = fp.readlines()
            parsed_lines = list(map(ast.literal_eval, lines))
            df = pd.DataFrame(parsed_lines)
            if ret_df:
                return df
            else:
                return df.columns.values, df.values

    @staticmethod
    def read_column(file, column, tl):
        with open(file, "r") as fp:
            lines = fp.readlines()
            parsed_lines = list(map(lambda x: tl(re.compile(f"'{column}': (.*?),").findall(x)[0]), lines))
            return parsed_lines

    @staticmethod
    def read_norms(file, itr):
        with open(file, "r") as fp:
            df = []
            lines = fp.readlines()
            for line in lines:
                norms = np.array(list(map(float, re.compile("gnorm.*\[(.*)\].*]").findall(line)[0].split(", "))))
                epoch = float(re.compile("\('epoch', (.*?)\)").findall(line)[0])
                df.append(np.column_stack([np.full_like(norms, itr), np.full_like(norms, epoch), norms])[:10])
            return np.concatenate(df, 0)

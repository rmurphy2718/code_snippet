from enum import Enum


class DetectableEnum(Enum):
    @classmethod
    def detect(cls, inp):
        return inp if type(inp) == cls else cls[inp]

    @classmethod
    def help(cls):
        return "|".join(map(str, cls))

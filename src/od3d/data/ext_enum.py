from enum import Enum

class ExtEnum(Enum):

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
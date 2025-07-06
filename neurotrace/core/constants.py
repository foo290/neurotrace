import enum
from typing import Literal


class Role():
    HUMAN = "human"
    AI = "ai"
    SYSTEM = "system"
    TOOL = "tool"

    # def __str__(self):
    #     return self.value

    @classmethod
    def from_string(cls, role_str: str):
        try:
            return cls[role_str.upper()]
        except KeyError:
            raise ValueError(f"Invalid role: {role_str}")

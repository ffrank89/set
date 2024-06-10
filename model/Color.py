from enum import Enum, auto

class ColorType(Enum):
    RED = 0
    PURPLE = 1
    GREEN = 2


class Color:

    def __init__(self, color) -> None:
        self.setColor(color)
    
    def __str__(self):
        return f"{self.color.name}"
    
    def __hash__(self):
        return hash(self.color)
    
    def __eq__(self, other):
        if not isinstance(other, Color):
            return NotImplemented
        return self.color == other.color

    def getColor(self):
        return self.color
    
    def setColor(self, color):
        if not isinstance(color, ColorType):
            raise ValueError("color must be given instance of ColorType enum")
        self.color = color
    

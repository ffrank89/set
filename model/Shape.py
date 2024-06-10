from enum import Enum, auto

class ShapeType(Enum):
    OVAL = auto()
    SQUIGGLE = auto()
    DIAMOND = auto()


class Shape:

    def __init__(self, shape) -> None:
        self.setShape(shape)
    
    def __str__(self):
        return f"{self.shape.name}"
    
    def __hash__(self):
        return hash(self.shape)
    
    def __eq__(self, other):
        if not isinstance(other, Shape):
            return NotImplemented
        return self.shape == other.shape

    def getShape(self):
        return self.shape
    
    def setShape(self, shape):
        if not isinstance(shape, ShapeType):
            raise ValueError("shape must be given instance of ShapeType enum")
        self.shape = shape
    




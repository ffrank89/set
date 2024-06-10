from enum import Enum, auto

class ShadingType(Enum):
    SOLID = auto()
    STRIPED = auto()
    OUTLINED = auto()


class Shading:

    def __init__(self, shading) -> None:
        self.setShading(shading)

    def __str__(self):
        return f"{self.shading.name}"
    
    def __hash__(self):
        return hash(self.shading)
    
    def __eq__(self, other):
        if not isinstance(other, Shading):
            return NotImplemented
        return self.shading == other.shading

    def getShading(self):
        return self.shading
    
    def setShading(self, shading):
        if not isinstance(shading, ShadingType):
            raise ValueError("shading must be given instance of ShadingType enum")
        self.shading = shading
    

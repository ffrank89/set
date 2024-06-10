from .Shape import Shape
from .Color import Color
from .Number import Number
from .Shading import Shading
import uuid
class Card:

    def __init__(self, shape: Shape, color: Color, number: Number, shading: Shading) -> None:
        self.shape = shape
        self.color = color
        self.number = number
        self.shading = shading
        self.id = uuid.uuid4()

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Card):
            return NotImplemented
        return (self.shape == other.shape and
                self.color == other.color and
                self.number == other.number and
                self.shading == other.shading)
    
    def __str__(self):
        return f"Shape: {self.shape.getShape()}\nColor: {self.color.getColor()}\nNumber: {self.number.getNumber()}\nShading: {self.shading.getShading()}"

    def __repr__(self):
        return f"Shape: {self.shape.getShape()}\nColor: {self.color.getColor()}\nNumber: {self.number.getNumber()}\nShading: {self.shading.getShading()}"

    def getShape(self):
        return self.shape.getShape()
    def getColor(self):
        return self.color.getColor()
    def getNumber(self):
        return self.number.getNumber()
    def getShading(self):
        return self.shading.getShading()
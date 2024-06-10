import unittest
from model.Shape import Shape, ShapeType
from model.Color import Color, ColorType
from model.Number import Number
from model.Shading import Shading, ShadingType
from model.Card import Card
from model.Computations import Computations

class TestIsSet(unittest.TestCase):

    def setUp(self):
        # Initialize example cards
        self.card1 = Card(Shape(ShapeType.OVAL), Color(ColorType.RED), Number(1), Shading(ShadingType.SOLID))
        self.card2 = Card(Shape(ShapeType.SQUIGGLE), Color(ColorType.GREEN), Number(2), Shading(ShadingType.STRIPED))
        self.card3 = Card(Shape(ShapeType.DIAMOND), Color(ColorType.PURPLE), Number(3), Shading(ShadingType.OUTLINED))
        self.card4 = Card(Shape(ShapeType.OVAL), Color(ColorType.RED), Number(1), Shading(ShadingType.SOLID))
        self.card5 = Card(Shape(ShapeType.OVAL), Color(ColorType.RED), Number(1), Shading(ShadingType.SOLID))

    def test_is_set_all_same(self):
        # Test where all properties are the same across all cards
        self.assertTrue(Computations.isSet(self.card4, self.card5, self.card4))

    def test_is_set_all_different(self):
        # Test where all properties are different across all cards
        self.assertTrue(Computations.isSet(self.card1, self.card2, self.card3))

    def test_is_set_not_a_set(self):
        # Test where not all properties meet the conditions of being all same or all different
        self.assertFalse(Computations.isSet(self.card1, self.card1, self.card2))

    def test_is_set_same_numbers_different_others(self):
        # Test where numbers are the same but other attributes are different
        card6 = Card(Shape(ShapeType.DIAMOND), Color(ColorType.PURPLE), Number(1), Shading(ShadingType.OUTLINED))
        card7 = Card(Shape(ShapeType.SQUIGGLE), Color(ColorType.GREEN), Number(1), Shading(ShadingType.STRIPED))
        self.assertTrue(Computations.isSet(self.card4, card6, card7))

    def test_is_set_mixed(self):
        # Test a mixed case that should not form a set
        card8 = Card(Shape(ShapeType.OVAL), Color(ColorType.GREEN), Number(1), Shading(ShadingType.OUTLINED))
        self.assertFalse(Computations.isSet(self.card1, card8, self.card2))

if __name__ == '__main__':
    unittest.main()
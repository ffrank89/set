import unittest
from model.Shape import Shape, ShapeType
from model.Color import Color, ColorType
from model.Number import Number
from model.Shading import Shading, ShadingType
from model.Card import Card
from model.Computations import Computations
from model.Board import Board

class TestGetAllSets(unittest.TestCase):

    def setUp(self):
        # Setup cards known to form a set
        self.card1 = Card(Shape(ShapeType.OVAL), Color(ColorType.RED), Number(1), Shading(ShadingType.SOLID))
        self.card2 = Card(Shape(ShapeType.SQUIGGLE), Color(ColorType.GREEN), Number(2), Shading(ShadingType.STRIPED))
        self.card3 = Card(Shape(ShapeType.DIAMOND), Color(ColorType.PURPLE), Number(3), Shading(ShadingType.OUTLINED))
        
        # Cards that do not form a set
        self.card4 = Card(Shape(ShapeType.OVAL), Color(ColorType.RED), Number(1), Shading(ShadingType.STRIPED))
        self.card5 = Card(Shape(ShapeType.OVAL), Color(ColorType.RED), Number(1), Shading(ShadingType.SOLID))

    def test_get_all_sets_with_valid_sets(self):
        board = Board()
        board.addCard(self.card1)
        board.addCard(self.card2)
        board.addCard(self.card3)
        all_sets = Computations.getAllSets(board)
        self.assertEqual(len(all_sets), 1)  # Expecting exactly one valid set
        expected_set = {self.card1, self.card2, self.card3}
        found_sets = [{card1, card2, card3} for card1, card2, card3 in all_sets]
        self.assertIn(expected_set, found_sets)

    def test_get_all_sets_with_no_valid_sets(self):
        board = Board()
        board.addCard(self.card1)
        board.addCard(self.card4)
        board.addCard(self.card5)
        all_sets = Computations.getAllSets(board)
        self.assertEqual(len(all_sets), 0)  # Expecting no sets

    def test_get_all_sets_with_multiple_possible_sets(self):
        board = Board()
        board.addCard(self.card1)
        board.addCard(self.card2)
        board.addCard(self.card3)
        board.addCard(self.card4)
        board.addCard(self.card5)
        # Assume card6 and card7 also form a set with card1
        card6 = Card(Shape(ShapeType.SQUIGGLE), Color(ColorType.GREEN), Number(3), Shading(ShadingType.OUTLINED))
        card7 = Card(Shape(ShapeType.DIAMOND), Color(ColorType.PURPLE), Number(2), Shading(ShadingType.SOLID))
        board.addCard(card6)
        board.addCard(card7)
        all_sets = Computations.getAllSets(board)
        for set in all_sets:
            print("SET: ", set)
        self.assertEqual(len(all_sets), 2)  # Expecting two sets

if __name__ == '__main__':
    unittest.main()

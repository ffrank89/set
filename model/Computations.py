from .Card import Card
from .Board import Board

from itertools import combinations
class Computations:
    
    @staticmethod
    def getAllSets(board: Board):
        if len(board.getBoard()) < 3:
            return []
        
        all_sets = set()
        card_combinations = combinations(board.getBoard(), 3)
        for combo in card_combinations:
            if Computations.isSet(combo[0], combo[1], combo[2]):
                all_sets.add(combo)
        return all_sets
    
    @staticmethod
    def isSet(card1: Card, card2: Card, card3: Card):

        return (Computations.checkProperty(card1.getShape(), card2.getShape(), card3.getShape()) and
                Computations.checkProperty(card1.getColor(), card2.getColor(), card3.getColor()) and
                Computations.checkProperty(card1.getNumber(), card2.getNumber(), card3.getNumber()) and
                Computations.checkProperty(card1.getShading(), card2.getShading(), card3.getShading()))

    @staticmethod
    def checkProperty(v1, v2, v3):
        return Computations.allEqual(v1, v2, v3) or Computations.allDifferent(v1, v2, v3)
    @staticmethod
    def allEqual(v1, v2, v3):
        return len(set([v1, v2, v3])) == 1
    @staticmethod
    def allDifferent(v1, v2, v3):
        return len(set([v1, v2, v3])) == 3
    
    
    


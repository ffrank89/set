from .Card import Card
class Board:
    def __init__(self) -> None:
        self.cards = set()

    def newBoard(self, cards):
        self.cards = set(cards)
    
    def addCard(self, card: Card):
        self.cards.add(card)

    def getBoard(self):
        return self.cards
        
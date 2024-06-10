class Number:

    def __init__(self, number: int) -> None:
        self.setNumber(number)
    
    def __str__(self):
        return f"{self.number}"
    
    def __hash__(self):
        return hash(self.number)
    
    def __eq__(self, other):
        if not isinstance(other, Number):
            return NotImplemented
        return self.number == other.number

    def getNumber(self):
        return self.number
    
    def setNumber(self, number):
        if number < 1 or number > 3:
            raise ValueError("Number must be between 1 and 3.")
        self.number = number
    

    

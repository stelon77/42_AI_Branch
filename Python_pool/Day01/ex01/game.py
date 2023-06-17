"""Lest's play game of Throne : live or die"""


class GotCharacter:
    """ Game of throne character to be inherited"""
    def __init__(self, first_name=None, is_alive=True):
        self.first_name = first_name
        self.is_alive = is_alive


class Tarly(GotCharacter):
    """
    A class representing the Tarly family.
    Or when reason triumphs over honor.
    """

    def __init__(self, first_name=None, is_alive=True):
        super().__init__(first_name=first_name, is_alive=is_alive)
        self.family_name = "Tarly"
        self.house_words = "First in battle"

    def print_house_words(self):
        """A method to spell the words of the house"""
        print(self.house_words)

    def die(self):
        """A method that kills the instance of the character"""
        self.is_alive = False


class Stark(GotCharacter):
    """
    A class representing the Stark family.
    Or when bad things happen to good people.
    """

    def __init__(self, first_name=None, is_alive=True):
        super().__init__(first_name=first_name, is_alive=is_alive)
        self.family_name = "Stark"
        self.house_words = "Winter is Coming"

    def print_house_words(self):
        """A method to spell the words of the house"""
        print(self.house_words)

    def die(self):
        """A method that kills the instance of the character"""
        self.is_alive = False


if __name__ == '__main__':
    arya = Stark("arya")
    print("Arya Alive ? ", arya.is_alive)
    arya.die()
    print("Arya Alive ? ", arya.is_alive)

    samwell = Tarly("Samwell")
    samwell.print_house_words()
    print(samwell.__dict__)
    print(samwell.__doc__)

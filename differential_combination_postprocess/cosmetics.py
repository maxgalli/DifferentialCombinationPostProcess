import itertools

markers = itertools.cycle(("v", "^", ">", "<", "s", "p", "*"))

class ListIterator:
    def __init__(self, ls):
        self.ls = ls
        self.idx = 0
    def __iter__(self):
        return self
    def __next__(self):
        try:
            return self.ls[self.idx]
        except IndexError:
            self.idx = 0
            return self.ls[self.idx]
        finally:
            self.idx += 1

# See https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
black_to_grey = ["k", "dimgrey", "darkgrey", "silver", "gainsboro"]
btg_iter = ListIterator(black_to_grey)

rainbow = ["red", "orange", "green", "blue", "indigo", "purple", "brown", "cyan", "pink", "olive"]
rainbow_iter = ListIterator(rainbow)
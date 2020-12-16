from collections import Counter
from itertools import chain, repeat, islice


def most_frequent(List): 
    counter=Counter(List)
    max_freq= max(set(List), key = List.count)
    return max_freq,counter[max_freq]


def CheckForGreater(list1, val):  
    return(all(x > val for x in list1))  

def pad_infinite(iterable, padding=None):
       return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
       return islice(pad_infinite(iterable, padding), size)

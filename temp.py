import sys


def is_valid(input):

    mapped_brackets = {'(':')', '{':'}', '[':']'}
    queue = []

    for char in input:
        if char in mapped_brackets.keys():
            queue.append(mapped_brackets[char])
        elif char in mapped_brackets.values():
            if len(queue)==0 or char != queue.pop():
                return False
    if len(queue)==0:
        return True
    return False

for line in sys.stdin:

    print(is_valid(line))










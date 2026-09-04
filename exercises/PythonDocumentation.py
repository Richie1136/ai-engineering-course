# WHY: Documentation is how we discover what an unfamiliar object can do and
# what inputs its methods expect. This small example creates two lists and uses
# `extend` so the method can be connected to its documented behavior: modify the
# first list by appending every element from another iterable.

list_1 = [1,2]
print(list_1)


list_2 = [34,45]
print(list_2)

list_1.extend(list_2)
print(list_1)

str()
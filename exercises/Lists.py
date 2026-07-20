# List - A type of sequence of data points


Participants = ["John", "Leila", "Gregory", "Cate"]

print(Participants[2])
print(Participants[-1])

Participants[-1] = "Robert"
print(Participants)
del Participants[2]
print(Participants)

# Append - Add to list

Participants.append("Joe")
Participants.extend(['Mike', "Sam"])

print(Participants)

print("The first participant is " + Participants[0])

# len() returns the number of elements in an object

print(len("Dolphin"))

print(len(Participants))

# List Slicing

# First number corresponds to the first position
# Second number is one posiiton above the last position we need

print(Participants[1:3])

print(Participants[:2]) # Grab first two values in the list

print(Participants[-2:]) # Grab last two values in the list


print(Participants.index("Joe"))

NewComers = ['Josh', 'Jim']
print(NewComers)

Bigger_list = [Participants, NewComers]

print(Bigger_list)

# Sort - sorts objects of list

Participants.sort()


print(Participants)
Participants.sort(reverse=True)
print(Participants)

Numbers = [10,40,100,30,70]

Numbers.sort()
print(Numbers)

Numbers.sort(reverse=True)
print(Numbers)

# Tuples - Are immutable(cannot be changed or modified) sequences

# Tuple's elements are placed within parentheses

# Functions can provide tuples as return values

x = (30,40,50)
print(x)

y = 50,80,100
print(y)

j,k,l = 40,60,80
print(l)

print(x[1])

List_1 = [x,y]
print(List_1)

(age, school_age) = "30,17".split(",")
print(age)
print(school_age)

def square_info(x):
    A = x ** 2
    P = 4 * x
    print("Area and Perimeter: ")
    return A,P

print(square_info(5))

def count(numbers):
    total = 0
    for x in numbers:
        if x < 20:
            total += 1
    return total

List_1 = [1,3,6,15,40,45,66,99, 17]

print(count(List_1))
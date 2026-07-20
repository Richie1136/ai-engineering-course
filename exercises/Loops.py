# iteration is the ability to execute a 
# certain code repeatedly

even = [0,2,4,6,8,10,12,14,16,18,20]

for i in even:  # i - is the loop variable
    print(i)

x = 0

while x <= 20:
    print(x)
    x += 2

# Range - range(start,stop,step) - 

# Creates a sequence of integers

# start = first number in the list
# stop = the last value + 1 (required)
# step = the distance between each two
# consecutive values on the list

range(10)
print(range(10)) # Start value 0, Step Value 1 and Stop value of 10

print(list(range(10))) # [0,1,2,3,4,5,6,7,8,9]

range(3,7)

print(list(range(3,7)))

print(list(range(1,20,2)))

# Iterating over Dictionaries


prices = {
    "box_of_spaghetti": 4,
    "lasagna": 5,
    "hamburger": 2
}

quantity = {
    "box_of_spaghetti": 6,
    "lasagna": 10,
    "hamburger": 0
}

money_spent = 0

for item in prices:
    money_spent += prices[item] * quantity[item]
print(money_spent)
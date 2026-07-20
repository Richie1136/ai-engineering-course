# List Comprehensions - One of Python's most distinguishable
# features

# List Comprehensions
# - Easy to understand
# - Quick to write
# - Support the option for setting conditionals either
# on the output or on the iterable or both
# - A very powerful tool (it can be applied to a very wide 
# range of cases and can deliver many types of output)
# - A fantastic example of high-quality code
# - Require more memory and run more slowly

numbers = [1,13,4,5,63,100]
print(numbers)

new_numbers = []

for n in numbers:
    new_numbers.append(n * 2)
print(new_numbers)

# List comprehensions can provide a concise way to build a 
# new list from the given "Numbers" list
new_numbers = [n * 2 for n in numbers]
# An output expression for an element in iterable
# iterable - the sequence we are going to iterate over

print(new_numbers)

for i in range(2):
    for j in range(5):
        print(i + j, end = " ")

new_list_comprehension_1 = [i + j for i in range(2) for j in range(5)]
print(new_list_comprehension_1)

print(type(new_list_comprehension_1))

new_list_comprehension_2 = [[i + j for i in range(2)] for j in range(5)]
print(new_list_comprehension_2)

print(type(new_list_comprehension_2))

print(type(new_list_comprehension_1[1]))
print(type(new_list_comprehension_2[1]))

print(list(range(1,11)))


for num in range(1,11):
    if (num % 2 != 0):
        print(num ** 3, end = " ")
# Deliver a list containing integers equal to the values of the
# generated sequence raised to the power of 3 on the condition that
# the base values are odd numbers
[num ** 3 for num in range(1,11) if num % 2 != 0]
[num ** 3 if num % 2 != 0 else "even" for num in range(1,11)]

# You can place a conditional on the right of the iterable to filter
# out certain values from the iterable like on line 53



products_on_sale = ["Chair_Type_1", "Chair_Type_2", "Chair_Type_3", "Chair_Type_4"]
sale_prices = [100,120,135, 150]
quantities = [1000,1500,1300]

for chair_type in products_on_sale:
    for price in sale_prices:
        for quantity in quantities:
            print ([chair_type, price*quantity])

sales_revenue = [[chair_type, price * quantity] for chair_type in products_on_sale for price in sale_prices for quantity in quantities]


print([j * 10 for j in range(1, 11) if j % 2 == 0])


new_list = []
quanties = [1,2,3,4,5,6,7,8,9,10]


for j in quanties:
    if (j % 2 == 0):
        new_list.append(j * 10)
print(new_list)


# for num in range(1,11):
#     if num % 2 == 0:
#         num * 10
#     else:
#         print("None")
        
print([num * 10 if num % 2 == 0 else "None" for num in range(1,11)])
# def - keyword


def simple():
    print("My first function")

simple()

def plus_ten(a):
    return a + 10


print(plus_ten(100))
print(plus_ten(10))

# return can be used only once in a function!


def plus_ten(a):
    result = a + 10
    print("Outcome")
    return result

print(plus_ten(2))

# print - does not affect the calculation of
# the output

# return - does not visualize the output
# It specifies what a certain function is supposed
# to give back


def wage(w_hours):
    return w_hours * 25

def with_bonus(w_hours):
    return wage(w_hours) + 50

print(wage(8))
print(with_bonus(8))

def add_ten(m):
    if m >= 100:
        return m + 10
    else:
        return "Save More"

print(add_ten(80))

def subtract_bc(a,b,c):
    result = a - b * c 
    print("Parameter a equals", a)
    print("Parameter b equals", b)
    print("Parameter c equals", c)
    return result
print(subtract_bc(10,3,2))

# Built in functions can be applied directly

# type() - obtains the type of variable you use as an argument

# int(), float(), str() transform their arguments in
# an integer, float, and string data type, respectively

print(type(1))
print(int(5.0))
print(float(3))
print(str(500))

# max() returns the highest value from a sequence of numbers

print(max(10,20,30))

# min() returns the lowest value from a sequence of numbers

print(min(10,20,30))

# abs() allows you to obtain the absolute value of its argument

z = -20
print(abs(z))

# sum() calculates the sum of all the elements in a list designated as an argument

list_1 = [1,2,3,4]
print(sum(list_1))

# round(x,y) returns the float of its argument (x), rounded to a specified number of 
# digits (y) after the decimal point

print(round(3.555555555555555555, 2))

print(round(3.2))

# pow(x,y) returns x to the power y

print(type(pow(2,10)))

# len() returns the number of elements in an object

print(len("Mathematics"))

# Lambda Functions in Python

def raise_to_the_power_of_2(x):
    return x ** 2

print(raise_to_the_power_of_2(3))

# Lambda expressions - Python's syntax for creating
# anonymous (Lambda) functions

# There are cases in which you will prefer not to define
# a whole new function(function you will only use once)

# Create a function that will allow you to focus on the 
# function itself

# A Lambda function is completely equivalent to an 
# ordinary function

# Lambda functions features

# - Can have one or many parameters but can contain a
# single expression only

# - Allow you to write just one line of code to include
# a simple functionality in a more complex expression

# - Can only be applied to the larger expression they have
# been written in

lambda x: x**2

raise_to_the_power_of_2_lambda = lambda x: x**2

print(raise_to_the_power_of_2_lambda(9))

# Parametrize x / 2 and pass an argument of 11
(lambda x: x / 2) (11)

print((lambda x: x / 2) (11))

(lambda x: (2 + 5 * x ** 4) ** 2 / (x + 3) ** 3) (2)
print((lambda x: (2 + 5 * x ** 4) ** 2 / (x + 3) ** 3) (2))

sum_xy = lambda x, y: x + y
print(sum_xy(4,6))

sum_xy = lambda x, y: x + y(x)
print(sum_xy(7, lambda x: x + 5))


product_xy = lambda x, y : x * y

print(product_xy(4,20))



print((lambda x: (135 - x ** 3) ** 4 / (1 + x) ** 5) (3))
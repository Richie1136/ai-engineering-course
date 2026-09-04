# WHY: Python gives different numeric and logical values different types. Type
# inspection and conversion matter because the operations allowed on an integer,
# float, string, or Boolean differ. `type()` observes; `int()` and `float()`
# create converted values.

x1 = 5
print(x1)

print(type(x1))
print(type(-6))
x2 = 4.75
print(type(x2))

print(int(x2))

print(float(5))

x3 = True

print(type(x3))
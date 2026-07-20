if (5 == 15 / 3):
    print("Hooray")

if (5 == 18 / 3):
    print("Hooray")
else:
    print("This is the else")


if (5 != 3 * 6):
    print("HipHop")


x = 10
y = 25

if x > 3 and y > 13:
    print("Both conditions are correct")
    
if x <= 3 or y <= 13:
    print("At least one of the conditions is false")


# Else statement

j = 1

if (j > 3):
    print("Case 1")
else:
    print("Case 2")

# The computer reads your commands from
# top to bottom!

def compare_to_five(r):
    if r > 5:
        return "Greater"
    elif r < 0:
        return "Negative"
    elif r < 5:
        return "Less"
    else:
        return "Equal" 

print(compare_to_five(-5))


x = 2

if x > 4:
    print("Correct")
else:
    print("Incorrect")


for n in range(10):
    print(2 ** n, end = " ")
print()

for x in range(20):
    if x % 2 == 0:
        print(x , end=" ")
    else:
        print("Odd")

x = [10,20,30]

for item in x:
    print(item)

for item in range(len(x)):
    print(x[item], end= " ")
# WHY: Indentation defines which statements belong to a function, conditional,
# or loop. Here, both assignment and return belong to `five`; the final print is
# outside the function and therefore runs only after the function is called.

def five(x):
    x = 5
    return x
print(five(300))
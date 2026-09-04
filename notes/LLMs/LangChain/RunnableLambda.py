from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import chain


# =====================================================
# Overview
# =====================================================

# So far, we've learned a lot about LangChain Expression Language and its
# underlying mechanisms.
#
# We discussed the Runnable class, which implements methods such as:
#
# - invoke()
# - batch()
# - stream()
#
# Runnable objects can be composed to create RunnableSequence objects,
# also known as chains.
#
# We can also invoke several runnables simultaneously using RunnableParallel.
#
# Another useful class is RunnablePassthrough, which allows values to pass
# through a chain without being modified.
#
# In this lesson, we'll learn how to transform ordinary functions into
# runnable objects using RunnableLambda.
#
# First, we'll create two simple lambda functions:
#
# - One that calculates the sum of the elements in a list
# - One that calculates the square of a number
#
# We'll then wrap them inside RunnableLambda objects so that we can call them
# using invoke().
#
# Finally, we'll combine the two runnables into a chain and graph the result.


# =====================================================
# Create Lambda Functions
# =====================================================

# Create a lambda function that calculates the sum of all elements in a list.

find_sum = lambda x: sum(x)

# print(find_sum([1, 2, 3, 4, 5]))

# 15


# Create a lambda function that calculates the square of a number.

find_square = lambda x: x**2

# print(find_square(7))

# 49


# =====================================================
# Convert Lambda Functions to Runnables
# =====================================================

# The functions we've created work normally, but they have one limitation.
#
# We can't pipe them into a LangChain chain because they aren't runnable
# objects and don't implement the invoke() method.
#
# To solve this, we can use RunnableLambda.


# Create a RunnableLambda that calculates the sum.

runnable_sum = RunnableLambda(
    lambda x: sum(x)
)

runnable_sum.invoke([1, 2, 3, 4, 5])

# We've now converted the lambda function into a runnable object that
# implements invoke().

# print(runnable_sum.invoke([1, 2, 3, 4, 5]))

# 15


# Create a RunnableLambda that calculates the square of a number.

runnable_square = RunnableLambda(
    lambda x: x**2
)

runnable_square.invoke(7)

# print(runnable_square.invoke(7))

# 49


# =====================================================
# Create the Chain
# =====================================================

# Pipe the two runnable objects together to create a RunnableSequence.

sum_square_chain = runnable_sum | runnable_square

# The first runnable calculates the sum of the numbers.
#
# The second runnable calculates the square of that result.

# print(sum_square_chain.invoke([1, 2, 3, 4, 5]))

# 225


# Essentially, we're doing:
#
# 1 + 2 + 3 + 4 + 5 = 15
#
# Then:
#
# 15 ** 2 = 225


# =====================================================
# Graph the Chain
# =====================================================

# Investigate what the graph of this chain looks like.

# print(sum_square_chain.get_graph().print_ascii())


# =====================================================
# Graph Output
# =====================================================

# We obtain four components.

# +-------------+
# | LambdaInput |
# +-------------+
#        *
#        *
#        *
#   +--------+
#   | Lambda |
#   +--------+
#        *
#        *
#        *
#   +--------+
#   | Lambda |
#   +--------+
#        *
#        *
#        *
# +--------------+
# | LambdaOutput |
# +--------------+


# =====================================================
# Graph Explanation
# =====================================================

# LambdaInput
#
# The first node represents the input:
#
# [1, 2, 3, 4, 5]


# First Lambda
#
# The first lambda calculates the sum:
#
# 1 + 2 + 3 + 4 + 5 = 15


# Second Lambda
#
# The output from the first lambda becomes the input to the second lambda.
#
# The second lambda calculates:
#
# 15 ** 2 = 225


# LambdaOutput
#
# The final node represents the output:
#
# 225


# =====================================================
# RunnableLambda with Regular Functions
# =====================================================

# We previously introduced LangChain's RunnableLambda class, which allows us
# to upgrade a function into a runnable object.
#
# We demonstrated this using Python's anonymous lambda functions, but we can
# wrap any regular function in the same way.


def find_sum(x):
    return sum(x)


def find_square(x):
    return x ** 2


# =====================================================
# Create a Chain Using Regular Functions
# =====================================================

# Define a chain using two instances of RunnableLambda connected with a
# pipe symbol.

chain1 = (
    RunnableLambda(find_sum)
    | RunnableLambda(find_square)
)

# We've recreated the previous example without using anonymous lambda
# functions.

# print(chain1.invoke([1, 2, 3, 4, 5]))

# 225


# =====================================================
# The @chain Decorator
# =====================================================

# The chain decorator wraps a function in a RunnableLambda, just as we
# manually did above.
#
# Creating runnables in this way is elegant and intuitive.
#
# As an added benefit, the function's name is also used as the name of the
# runnable.
#
# When using the chain decorator in your code, avoid naming your variables
# "chain".
#
# Doing so introduces a naming conflict that could result in an error.
#
# A Python decorator is simply a function.
#
# Decorators modify the behavior of another function or class, enhancing
# its functionality.


@chain
def runnable_sum(x):
    return sum(x)


@chain
def runnable_square(x):
    return x ** 2


# =====================================================
# Check the Runnable Types
# =====================================================

# The decorated functions are instances of the RunnableLambda class.

# print(type(runnable_sum))

# <class 'langchain_core.runnables.base.RunnableLambda'>


# print(type(runnable_square))

# <class 'langchain_core.runnables.base.RunnableLambda'>


# =====================================================
# Create a Chain Using Decorated Functions
# =====================================================

chain2 = (
    runnable_sum
    | runnable_square
)

# print(chain2.invoke([1, 2, 3, 4, 5]))

# 225


# =====================================================
# Summary
# =====================================================

# RunnableLambda allows us to convert a regular Python function into a
# runnable object.
#
# Once converted, the function can:
#
# - Use invoke()
# - Be added to a chain
# - Pass its output to another runnable
#
# This allows ordinary Python functions to become components inside
# LangChain Expression Language chains.
#
# We can create RunnableLambda objects in two ways:
#
# - Wrap a function manually with RunnableLambda()
# - Use the @chain decorator
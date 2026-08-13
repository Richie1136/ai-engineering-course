# Bert calculates where the answer to our question is within our text by creating start vectors and end vectors.

# It then takes the start and end vecotrs and takes the dot product between these and the final embedding for each token.

# This output then runs through a softmax function to give a probability score. The word with the highest probability is considered
# the correct start or end token, respectively.
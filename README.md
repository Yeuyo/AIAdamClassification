# AIAdamClassification
Machine Learning Classification using Adam Optimizer.

The code is written using numpy library to achieve the following:
- Randomized initial weights (lines 17-23).
- Created a sigmoid layer for forward/backward pass (lines 25-29).
- Trained NN with Adam's optimizer.
- Sample input (need to change this to make use of real data, lines 54-68).
- Stopping criteria (line 71 and 72).
- Customisable hidden layers numbers and nodes (line 73).
- The code currently test the trained network on input data, this should be changed to test it on test data (lines 120-126).
- The code also included a cost function to calculate the cost and see whether the optimizer is moving in the correct direction, but it was removed after it was tested to be working (lines 41-51), the cost at each iteration/epoch can be computed using the function "costFunction()".

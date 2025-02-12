import numpy as np
import matplotlib.pyplot as plt

# --------------------- Part A -------------------------


# initialize function as a function for later
def myfunc(theta):
    return 36 * theta - 8 * (theta - 1) ** 2 + (theta - 3) ** 4


# initialize range
x = np.linspace(0, 6, 100)
# initialize function
y = myfunc(x)

# create line plot
plt.plot(x, y, label="y = 36x − 8(x − 1)^2 + (x − 3)^4")
plt.title("Question 1 ECE601 2025 Aidan Chin\nPlot of y = f(x)")
plt.ylabel("y")
plt.xlabel("x")
plt.grid(True)
plt.show()

# --------------------- Part B -------------------------
# Gradient descent will find only the local minimum,
# so if there are 2 points where there are different
# local minimums, it will find the one it is closest to

# --------------------- Part C -------------------------
# implement gradient descent
# looking at the outputs from both times we ran the descent
#  algorithm, an x value of about .88 has a y value lower than
#  an x value of about 4.8: 51 < 67


# derivative function (gradient)
def gradient(theta):
    return 36 - 16 * (theta - 1) + 4 * (theta - 3) ** 3


# descent algorithm
def gradient_descent(starting_theta, learning_rate, iterations):
    theta = starting_theta
    print("Starting Descent at theta =", theta)
    for _ in range(iterations):
        grad = gradient(theta)
        theta -= learning_rate * grad
    print(f"Theta = {theta}, Cost = {myfunc(grad)}")
    print("Optimizing Complete")
    return theta


# initialize parameters
starting_theta = 1
learning_rate = 0.001
iterations = 1000

# run algorithm
x = gradient_descent(starting_theta, learning_rate, iterations)
print(f"Local Minimum 1 at ({x}, {myfunc(x)})")

# update starting position
starting_theta = 5.9

# run algorithm again
x = gradient_descent(starting_theta, learning_rate, iterations)
print(f"Local Minimum 2 at ({x}, {myfunc(x)})")

# --------------------- Part C -------------------------
# If I am understanding the question correctly, it is asking
# how do we know which side of the local max the GD algorithm
# will choose based on just the initial values. To answer that
# question, we just look at the gradient part of the algorithm,
# which is what we iterate with. This is simply the derivative
# at the point we are working from so if the initial position
# is on the left side of the local max, the derivative will be
# positive, and because we subtract that from the initial point,
# we move the opposite direction from the local max, the same can
# be said for the on the other side of the local max, where the
# derivative will be negative, subtracting a negative increases
# and moves away from the local max again, working to find the
# local minimum in the other direction. if we start exactly on
# the local max, a funny thing happens where the derivative is
# zero and it will not move in either direction. in this edge
# case, the algorithm fails to find a local minimum.

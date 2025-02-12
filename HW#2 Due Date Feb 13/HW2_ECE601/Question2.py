import numpy as np
import matplotlib.pyplot as plt

# modify function from Question 1 to run with new cost function and output a cost array:


# original cost function
def myfunc(theta):
    return (theta - 2) ** 2


# derivative function (gradient)
def gradient(theta):
    return 2 * theta - 4


# descent algorithm
def gradient_descent(starting_theta, learning_rate, iterations):
    theta = starting_theta
    costarray = [myfunc(starting_theta)]
    print("Starting Descent at theta =", theta)
    for _ in range(iterations):
        grad = gradient(theta)
        theta -= learning_rate * grad
        costarray.append(myfunc(theta))
    print(f"Theta = {theta}, Cost = {myfunc(theta)}")
    print("Optimizing Complete")
    return costarray


def learning():
    starting_theta = 0
    iterations = 10

    # run GD with different learning rates collecting cost array
    y1 = gradient_descent(starting_theta, 0.01, iterations)
    y2 = gradient_descent(starting_theta, 0.1, iterations)
    y3 = gradient_descent(starting_theta, 1.01, iterations)
    x = np.linspace(0, iterations, iterations + 1)

    plt.plot(x, y1, label="a = .01")
    plt.plot(x, y2, label="a = .1")
    plt.plot(x, y3, label="a = 1.01")
    plt.legend(
        title="Learning Rate (a)",
        loc="best",
        fontsize=10,
        title_fontsize=11,
        frameon=True,
    )
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title(
        "Question 2 ECE601 2025 Aidan Chin\nGradient Descent Progress for Different Learning Rates"
    )
    plt.grid(True)
    plt.show()


learning()

# here this is a trade-off with learning rate,
# where a really low number takes forever to
# converge while a large one is quicker, but
# go too large and youll skip right over and
# never covverge because it only gets further
# and further as each jump only gets bigger.
# at a learning rate of .01, the alg is just 
# too slow and doesnt converge. 
# at a learning rate of .1 it gets a lot closer but still in 
# the allowed iterations, it doesnt converge. 
# at a learning rate of 1.01, it jumps over 
# the local min and diverges.
import numpy as np

print("Problem 1")

def problem1(A, B):
	return A + B

# Testing the function with an example
x = np.array([[1, 2, 1], [0, -1, 5]])
y = np.array([[11, -1, 5], [2, 3, 1]])
problem1_result = problem1(x, y)	
print(problem1_result)

print('=' * 80)

print("Problem 2")
def problem2(A, B, C):
    """
    Compute AB - C, where AB is the matrix multiplication of A and B, and then subtract matrix C.

    :param A: A numpy matrix
    :param B: A numpy matrix
    :param C: A numpy matrix
    :return: The result of AB - C
    """
    # Compute the matrix multiplication AB
    AB = np.dot(A, B)
    
    ABC = AB - C
    # Compute and return the final result
    return ABC

# Testing the function with an example
x = np.array([[1, 2, 1], [0, -1, 5]])
y = np.array([[11, -1, 5], [2, 3, 1], [2, 1, 0]])
z = np.array([[0, -2, -1], [2, 1, 0]])
problem2_result = problem2(x, y, z)	
print(problem2_result)

print('=' * 80)

print("Problem 3")
def problem3(A, B, C):
    """
    Compute the element-wise (Hadamard) product of A and B, and add the transpose of C.

    :param A: A numpy matrix
    :param B: A numpy matrix
    :param C: A numpy matrix
    :return: The result of (A ⊙ B) + C^T
    """
    # Compute the Hadamard product
    hadamard_product = A * B
    
    # Compute the transpose of C
    C_transpose = np.transpose(C)
    
    # Compute and return the final result
    return hadamard_product + C_transpose

# Testing the function with an example
A3 = np.array([[1, 2, 1], [0, -1, 5]])
B3 = np.array([[11, -1, 5], [2, 3, 1]])
C3 = np.array([[0, -2], [2, -1], [2, 1]])
problem3_result = problem3(A3, B3, C3)	
print(problem3_result)

print('=' * 80)

print("Problem 4")

def problem4(A):
    """
    Create a zero matrix with the same dimensions as matrix A.

    :param A: A numpy matrix
    :return: A zero matrix with the same dimensions as A
    """

    shape = (A.shape[0], A.shape[1])  # tuple of dimensions

    # Create and return the zero matrix
    return np.zeros(shape)

# Testing the function with an example
A4 = np.array([[1, 2, 1], [0, -1, 5]])
problem4_result = problem4(A4)	
print(problem4_result)

print('=' * 80)

print("Problem 5")

def problem5(A, x):
    """
    Compute A^{-1}x for square matrix A and column vector x using np.linalg.solve.
    
    :param A: A square numpy matrix
    :param x: A column vector
    :return: The result of A^{-1}x
    """
    # Compute A^{-1}x using np.linalg.solve
    return np.linalg.solve(A, x)

# Testing the function with an example
A7 = np.array([[1, 2], [-1, 5]])
x7 = np.array([[1.], [5.]])
problem5_result = problem5(A7,x7)	
print(problem5_result)

print('=' * 80)

print("Problem 6")

def problem6(A, i):
    """
    Compute the sum of all the entries in the ith row of matrix A.

    :param A: A numpy matrix
    :param i: The index of the row (0-indexed)
    :return: Sum of the entries in the ith row of A
    """
    # Compute and return the sum of the ith row
    return np.sum(A[i])

# Testing the function with an example
A6 = np.array([[1, 2, 5], [-1, 5, 0]])
i6 = 1 # or input("Please type 1 or 2:")
problem6_result = problem6(A6, i6)	
print(problem6_result)

print('=' * 80)

print("Problem 7")

def problem7(A, c, d):
    """
    Compute the arithmetic mean of all entries in matrix A that are between c and d (inclusive).
    
    :param A: A numpy matrix
    :param c: Lower bound (inclusive)
    :param d: Upper bound (inclusive)
    :return: Arithmetic mean of the entries of A that lie between c and d
    """
    # Find the indices of elements that are between c and d (inclusive)
    indices = np.nonzero(A)

    # Extract the elements from A that lie in the specified range
    elements_in_range = np.mean(indices[c:d])

    # Compute and return the mean of these elements
    return ...

# Testing the function with an example
A7 = np.array([[1, 2, 5], [-1, 5, 0], [4, 5, 9]])
i7 = 0 # or input("Please insert scalar c:")
j7 = 5 # or input("Please insert scalar d > c:")
problem7_result = problem7(A7, i7, j7)	
print(problem7_result)

print('=' * 80)

print("Problem 8")
def problem8(x, k, m, s):
    """
    Generate samples from a multidimensional Gaussian distribution.

    :param x: An n-dimensional column vector (numpy array of shape (n, 1))
    :param k: An integer specifying the number of samples to generate
    :param m: A positive scalar
    :param s: A positive scalar
    :return: An n x k matrix, each column of which is a sample from the distribution
    """
	# Determine the dimension of column vector
    n = ...
    
    # Generate n-dimensional column vector with all ones
    z = ...
    
    # Calculate mean for the distribution
    mean = ...
    
    # Create covariance matrix, sI, where I is the identity matrix
    cov = ...
    
    # Generate k samples from the multivariate normal distribution
    samples = ...
    
    return samples

# Set the random seed for reproducibility
np.random.seed(42)  # You can use any number here

# Testing the function with an example
X8 = np.array([[1], [2], [5], [-1], [5], [0], [4], [5], [9]])
k8 = 6 # or input("Please insert integer k:")
m8 = 1 # or input("Please insert positive scalar m:")
s8 = 3 # or input("Please insert positive scalar s:")
problem8_result = problem8(X8, k8, m8, s8)	
print(problem8_result)

print('=' * 80)

print("Problem 9")
import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

def problem9(filepath):
    # Read the CSV file into a pandas DataFrame
    ...
    
    # Print the shape of the DataFrame
    ...
    
    # Print the first column. 
    ...
    
    # Print the first row. 
    ...

# Example usage:
# Please use the CSV file uploaded on Canvas, namely 'Lec1_50_Startups.csv.' You can find it in the Lectures module.
# Please replace 'path_to_your_file' with the actual path of the dataset on your Google Drive.
problem9('path_to_your_file.csv')
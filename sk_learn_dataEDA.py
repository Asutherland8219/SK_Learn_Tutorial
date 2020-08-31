### Machine learning tutorial on Sci Kit learn from Data camp; tutorial can be found here : https://www.datacamp.com/community/tutorials/machine-learning-python ###
from sklearn import datasets
import pandas as pd

# load the data 
digits = datasets.load_digits()

# Display the data set 
print(digits)

# Explore the data set 
print(digits.keys())

# Print data 
print(digits.data)

# Print out the target values
print(digits.target)

# Print ouf the description
print(digits.DESCR)


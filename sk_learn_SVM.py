### Trying out another model; Support Vector Machines ###
### Machine learning tutorial on Sci Kit learn from Data camp; tutorial can be found here : https://www.datacamp.com/community/tutorials/machine-learning-python ###
from sklearn import datasets
from sklearn import svm
import numpy as np

# Preprocessing tools
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

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

## Because this data is numpy array; the data is not sorted in an excel sheet format. It is important to know about the shape of the array to accurately handle the data. ##
# WE must seperate the data into a more usuable format #

# Isolate the digits data 
digits_data = digits.data

# Inspect
print(digits_data.shape)

# Isolate target values with the function target
digits_target = digits.target

# Inspect the shape of the new variable
print(digits_target.shape)

# Print the number of unique labels 
number_digits = len(np.unique(digits.target))

# Isolate the 'images' variable
digits_images = digits.images

# Inspect the shape of the digits set
print(digits_images.shape)

# You notice in the results that in the images set they are 8px by 8px. However, the target data has 64 features according to our inspection. We can reshape the data in order to visualize it. #
print(np.all(digits.images.reshape((1797,64)) == digits.data))

# Split the data into training and test sets
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(digits.data, digits.target, digits.images, test_size=0.25, random_state=42)

# Create the SVC Model 
svc_model = svm.SVC(gamma=0.001, C=100., kernel='linear')

# Fit the data to the svc model 
svc_model.fit(X_train, y_train)
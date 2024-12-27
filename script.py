import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize


def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
  
    # Adding bias term
    train_data = np.hstack((np.ones((n_data, 1)), train_data))
    #print(len(train_data))
    n_data = train_data.shape[0]

    # Compute sigmoid of weighted inputs
    weights_transpose = initialWeights.reshape(-1, 1)
    z = np.dot(train_data, weights_transpose)
    theta = sigmoid(z)

    # Error computation
    error = np.dot(labeli.transpose(),np.log(theta)) + np.dot(np.subtract(1.0,labeli).transpose(),np.log(np.subtract(1.0,theta)))
    error = np.sum(error)
    error = (-error)/n_data
    #print("Error:", error)
    
    # Gradient of error function computation
    error_grad = np.dot(train_data.transpose(),np.subtract(theta,labeli))
    error_grad = error_grad/n_data
    #print("Error gradient:", error_grad)

    return error, error_grad.flatten()


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    
    # Adding bias term
    n_data = data.shape[0]
    data = np.hstack((np.ones((n_data, 1)), data))
    probabilities = sigmoid(np.dot(data, W))

    # Predicting class with maximum probability
    label = np.argmax(probabilities, axis=1).reshape((n_data, 1))
    return label

def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    
    """

    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))
    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    initial_weights = params.reshape((n_feature + 1, labeli.shape[1]))

    # Adding bias term
    train_data_new = np.hstack([np.ones((train_data.shape[0], 1)), train_data])

    # Computing the class probabilities using softmax
    logits = np.dot(train_data_new, initial_weights)
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Error computation
    log_softmax = np.log(softmax_probs)
    error = -np.sum(labeli * log_softmax) / n_data

    # Gradient of error function computation
    error_grad = np.dot(train_data_new.T, (softmax_probs - labeli)) / n_data
    #print(error)
    return error, error_grad.flatten()
    



def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    n_data = data.shape[0]

    # Adding bias term
    bias = np.ones((n_data, 1))
    data_with_bias = np.hstack([bias, data])

    # Computing class probabilities using softmax
    logits = np.dot(data_with_bias, W)
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    labels = np.argmax(softmax_probs, axis=1).reshape(-1, 1)
    return labels


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
initialWeights = initialWeights.flatten()
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Finding the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Finding the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Finding the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')

from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Train and evaluate using Linear Kernel
train_label_flattened = np.squeeze(train_label)
linear_svm = SVC(kernel='linear')
linear_svm.fit(train_data, train_label_flattened)
print("Linear Kernel fitting completed")

train_acc_linear = linear_svm.score(train_data, train_label_flattened)
print(f"Training accuracy (Linear Kernel): {train_acc_linear:.2f}")

test_acc_linear = linear_svm.score(test_data, test_label)
print(f"Testing accuracy (Linear Kernel): {test_acc_linear:.2f}")

validation_acc_linear = linear_svm.score(validation_data, validation_label)
print(f"Validation accuracy (Linear Kernel): {validation_acc_linear:.2f}")
print("\n-------------------------------\n")

# Train and evaluate using RBF Kernel with gamma set to 1
rbf_svm_1 = SVC(gamma=1)
rbf_svm_1.fit(train_data, train_label_flattened)
print("RBF Kernel with gamma = 1 fitting completed")

train_acc_rbf1 = rbf_svm_1.score(train_data, train_label_flattened)
print(f"Training accuracy (RBF Kernel, gamma = 1): {train_acc_rbf1:.2f}")

test_acc_rbf1 = rbf_svm_1.score(test_data, test_label)
print(f"Testing accuracy (RBF Kernel, gamma = 1): {test_acc_rbf1:.2f}")

validation_acc_rbf1 = rbf_svm_1.score(validation_data, validation_label)
print(f"Validation accuracy (RBF Kernel, gamma = 1): {validation_acc_rbf1:.2f}")
print("\n-------------------------------\n")

# Train and evaluate using RBF Kernel with default gamma
rbf_svm_default = SVC(kernel='rbf')
rbf_svm_default.fit(train_data, train_label_flattened)
print("RBF Kernel with default gamma fitting completed")

train_acc_rbf_default = rbf_svm_default.score(train_data, train_label_flattened)
print(f"Training accuracy (RBF Kernel, default gamma): {train_acc_rbf_default:.2f}")

test_acc_rbf_default = rbf_svm_default.score(test_data, test_label)
print(f"Testing accuracy (RBF Kernel, default gamma): {test_acc_rbf_default:.2f}")

validation_acc_rbf_default = rbf_svm_default.score(validation_data, validation_label)
print(f"Validation accuracy (RBF Kernel, default gamma): {validation_acc_rbf_default:.2f}")
print("\n-------------------------------\n")

# Experimenting with varying C values for RBF Kernel
C_values = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
train_accuracies = []
test_accuracies = []
validation_accuracies = []

# Loop through each C value
for C_val in C_values:
    svm_rbf_with_C = SVC(C=C_val)
    print(f"Fitting SVM with C = {C_val}")
    svm_rbf_with_C.fit(train_data, train_label_flattened)
    print("Model fitting completed")

    train_acc = svm_rbf_with_C.score(train_data, train_label_flattened)
    test_acc = svm_rbf_with_C.score(test_data, test_label)
    validation_acc = svm_rbf_with_C.score(validation_data, validation_label)

    print(f"Training accuracy (C={C_val}): {train_acc:.5f}")
    print(f"Testing accuracy (C={C_val}): {test_acc:.5f}")
    print(f"Validation accuracy (C={C_val}): {validation_acc:.5f}")
    print("\n-------------------------------\n")

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    validation_accuracies.append(validation_acc)

# Converting accuracy to percentage
train_accuracies_percent = [acc * 100 for acc in train_accuracies]
test_accuracies_percent = [acc * 100 for acc in test_accuracies]
validation_accuracies_percent = [acc * 100 for acc in validation_accuracies]

# Plot accuracy vs C values
accuracy_matrix = np.column_stack((train_accuracies_percent, test_accuracies_percent, validation_accuracies_percent))
plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(C_values, accuracy_matrix)
plt.title('Accuracy vs. C Values for RBF Kernel')
plt.xlabel('C values')
plt.ylabel('Accuracy (%)')
plt.legend(['Training Accuracy', 'Testing Accuracy', 'Validation Accuracy'], loc='best')
plt.show()



"""
Script for Extra Credit Part
"""

# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = initialWeights_b.flatten()
opts_b = {'maxiter': 100}


args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
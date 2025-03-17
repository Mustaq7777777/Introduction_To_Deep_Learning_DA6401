#!/usr/bin/env python
import argparse
import wandb
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist, mnist

##########################
# Activation Functions   #
##########################

def softmax(v):
    # Compute softmax in a numerically stable way by subtracting the max value.
    exp_vector = np.exp(v - np.max(v))
    return exp_vector / np.sum(exp_vector)

# Sigmoid activation function.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function.
def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Hyperbolic tangent activation function.
def tanh(x):
    return np.tanh(x)

# Derivative of the tanh function.
def d_tanh(x):
    return 1 - np.tanh(x) ** 2

# Rectified Linear Unit (ReLU) activation function.
def relu(x):
    return np.maximum(0, x)

# Derivative of the ReLU function.
def d_relu(x):
    return np.where(x > 0, 1, 0)

# Function to create a one-hot encoded vector.
def e(l, length):
    y = np.zeros([length, 1])
    y[int(l)] = 1
    return y




##########################
# Weight Initialization  #
##########################

def init(W, b, input_nodes, hiddenlayers, hiddennodes, output_nodes, initializer):
    # Set a seed for reproducibility.
    np.random.seed(1)
    
    if initializer == "Xavier":
        # Xavier initialization for the first hidden layer (input to hidden).
        W[1] = np.random.normal(0.0, np.sqrt(1.0 / input_nodes), (hiddennodes, input_nodes))
        b[1] = np.zeros((hiddennodes, 1))
        # Xavier initialization for subsequent hidden layers.
        for i in range(2, hiddenlayers + 1):
            W[i] = np.random.normal(0.0, np.sqrt(1.0 / hiddennodes), (hiddennodes, hiddennodes))
            b[i] = np.zeros((hiddennodes, 1))
        # Xavier initialization for the output layer (hidden to output).
        W[hiddenlayers + 1] = np.random.normal(0.0, np.sqrt(1.0 / hiddennodes), (output_nodes, hiddennodes))
        b[hiddenlayers + 1] = np.zeros((output_nodes, 1))
    elif initializer == "random":
        # Random initialization for the first hidden layer.
        W[1] = np.random.rand(hiddennodes, input_nodes) - 0.5
        b[1] = np.zeros((hiddennodes, 1))
        # Random initialization for subsequent hidden layers.
        for i in range(2, hiddenlayers + 1):
            W[i] = np.random.rand(hiddennodes, hiddennodes) - 0.5
            b[i] = np.zeros((hiddennodes, 1))
        # Random initialization for the output layer.
        W[hiddenlayers + 1] = np.random.rand(output_nodes, hiddennodes) - 0.5
        b[hiddenlayers + 1] = np.zeros((output_nodes, 1))

    return W, b



##########################
# Forward Propagation    #
##########################

# Applies the selected activation function to the input.
def g(z, act_func):
    if act_func == "sigmoid":
        return sigmoid(z)
    elif act_func == "tanh":
        return tanh(z)
    elif act_func == "relu":
        return relu(z)

# Applies the derivative of the selected activation function.
def g_derivative(z, act_func):
    if act_func == "sigmoid":
        return d_sigmoid(z)
    elif act_func == "tanh":
        return d_tanh(z)
    elif act_func == "relu":
        return d_relu(z)

# Computes the output layer activation using softmax.
def output_activation(z, act_func):
    if act_func == "softmax":
        return softmax(z)

# Performs forward propagation for a single training example.
def forward_propagation(x_data, hidden_layers, hidden_nodes, input_nodes, output_nodes, W, b, a, h, L, act_func):
    # Set input layer activation by reshaping the L-th training example.
    h[0] = x_data[L].reshape(-1, 1)
    # Forward pass through all hidden layers.
    layer = 1
    while layer <= hidden_layers:
        # Compute linear combination: weighted sum + bias.
        a[layer] = np.dot(W[layer], h[layer - 1]) + b[layer]
        # Apply activation function.
        h[layer] = g(a[layer], act_func)
        layer += 1
    # Compute output layer activation.
    a[hidden_layers + 1] = np.dot(W[hidden_layers + 1], h[hidden_layers]) + b[hidden_layers + 1]
    ycap = output_activation(a[hidden_layers + 1], "softmax")
    return ycap, a, h



##########################
# Back Propagation       #
##########################

def back_propagation(y_train, hiddenlayers, hiddennodes, input_nodes, output_nodes,
                     W, b, a, h, gradient_weights, gradient_biases, sample_idx, yhat, activation_func, loss_func):

    output_layer = hiddenlayers + 1
    delta = {}

    # Compute the error (delta) for the output layer based on the chosen loss function.
    if loss_func == "mean_squared_error":
        target = e(y_train[sample_idx].item(), output_nodes)
        # Calculate Jacobian for the softmax output.
        jacobian = np.diagflat(yhat) - np.dot(yhat, yhat.T)
        # Chain rule application and scaling by the number of output nodes.
        delta[output_layer] = np.dot(jacobian, (yhat - target)) / output_nodes
    elif loss_func == "cross_entropy":
        delta[output_layer] = yhat - e(y_train[sample_idx].item(), output_nodes)
    


    # Propagate the error backward through the network.
    k = output_layer
    while k > 0:
        # Compute gradients for weights and biases.
        grad_W = np.dot(delta[k], h[k-1].T)
        grad_b = delta[k]
        # Accumulate gradients in gradient_weights and gradient_biases dictionaries.
        if k in gradient_weights:
            gradient_weights[k] += grad_W
            gradient_biases[k] += grad_b
        else:
            gradient_weights[k] = grad_W
            gradient_biases[k] = grad_b
        # Backpropagate the error to the previous layer.
        if k > 1:
            prev_error = np.dot(W[k].T, delta[k])
            delta[k-1] = np.multiply(prev_error, g_derivative(a[k-1], activation_func).reshape(-1, 1))
        k -= 1

    return gradient_weights, gradient_biases

# Computes the loss for a single training example.
def loss(yhat, y_train, loss_func, sample_idx, output_nodes):

    if loss_func == "mean_squared_error":
        target = e(y_train[sample_idx].item(), output_nodes)
        return np.mean(np.square(yhat - target))
    elif loss_func == "cross_entropy":
        return -np.log(yhat[y_train[sample_idx].item()])




##########################
# Model Evaluation       #
##########################

def evaluate_performance_train(x_train, y_train, hidden_layers, hidden_nodes, input_nodes, output_nodes,
                               W, b, activation_func, loss_func):
    # Evaluate training performance on the first 54000 samples.
    correct_train = 0
    total_train_loss = 0
    # going over training examples
    for idx in range(0, 54000):
        yhat, _, _ = forward_propagation(x_train, hidden_layers, hidden_nodes, input_nodes,
                                         output_nodes, W, b, {}, {}, idx, activation_func)
        total_train_loss += loss(yhat, y_train, loss_func, idx, output_nodes)
        pred = int(np.argmax(yhat))
        if pred == y_train[idx]:
            correct_train += 1
    
    # accuracy and loss calculation
    train_accuracy = (correct_train / 54000) * 100
    train_loss = total_train_loss / 54000
    wandb.log({"train_accuracy": train_accuracy, "train_loss": train_loss})
    return train_accuracy, train_loss

def evaluate_performance_validity(x_train, y_train, hidden_layers, hidden_nodes, input_nodes, output_nodes,
                                  W, b, activation_func, loss_func):
    # Evaluate validation performance on the last 6000 training samples.
    correct_val = 0
    total_val_loss = 0

    # going over validation dataset
    for idx in range(54000, 60000):
        yhat, _, _ = forward_propagation(x_train, hidden_layers, hidden_nodes, input_nodes,
                                         output_nodes, W, b, {}, {}, idx, activation_func)
        total_val_loss += loss(yhat, y_train, loss_func, idx, output_nodes)
        pred = int(np.argmax(yhat))
        if pred == y_train[idx]:
            correct_val += 1
    
    # accuracy and loss calculation
    val_accuracy = (correct_val / (60000 - 54000)) * 100
    val_loss = total_val_loss / (60000 - 54000)
    wandb.log({"val_accuracy": val_accuracy, "val_loss": val_loss})
    return val_accuracy, val_loss

def evaluate_performance_test(x_test, y_test, hidden_layers, hidden_nodes, input_nodes, output_nodes,
                              W, b, activation_func):
    # Evaluate test accuracy.
    correct_test = 0

    # going over test dataset
    for idx in range(len(x_test)):
        yhat, _, _ = forward_propagation(x_test, hidden_layers, hidden_nodes, input_nodes,
                                         output_nodes, W, b, {}, {}, idx, activation_func)
        pred = int(np.argmax(yhat))
        if pred == y_test[idx]:
            correct_test += 1
    
    # accuracy calculation
    test_accuracy = (correct_test / len(x_test)) * 100
    wandb.log({"test_accuracy": test_accuracy})
    return test_accuracy



##########################
# Optimizers             #
##########################

# Stochastic Gradient Descent (SGD) optimizer :

def SGD(x_train, y_train, hiddenlayers, hiddennodes, input_nodes, output_nodes,
        W, b, a, h, learning_rate, num_epochs, batch_size, train_set,
        activation_func, loss_func, weight_decay):
    
    for epoch in range(num_epochs):
        # Initialize gradient accumulators.
        gradient_weights = {}
        gradient_biases = {}
        # Loop over all training samples.
        for index in range(train_set):
            # Forward propagation for current sample.
            yhat, a, h = forward_propagation(x_train, hiddenlayers, hiddennodes, input_nodes,
                                             output_nodes, W, b, a, h, index, activation_func)
            # Compute gradients via backpropagation.
            back_propagation(y_train, hiddenlayers, hiddennodes, input_nodes, output_nodes,
                             W, b, a, h, gradient_weights, gradient_biases, index, yhat, activation_func, loss_func)
            # Update weights in batches.
            if index % batch_size == 0:
                i = 1
                while i < hiddenlayers + 2:
                    W[i] -= learning_rate * (gradient_weights[i] + weight_decay * W[i])
                    b[i] -= learning_rate * gradient_biases[i]
                    i += 1
                # Reset gradient accumulators after each batch.
                gradient_weights, gradient_biases = {}, {}
        # Log epoch number.
        wandb.log({"epoch " : epoch})

        # Evaluate and log performance on test, validation, and training sets.
        evaluate_performance_train(x_train, y_train, hiddenlayers, hiddennodes, input_nodes, output_nodes, W, b, activation_func, loss_func)
        evaluate_performance_validity(x_train, y_train, hiddenlayers, hiddennodes, input_nodes, output_nodes, W, b, activation_func, loss_func)
        evaluate_performance_test(x_test, y_test, hiddenlayers, hiddennodes, input_nodes, output_nodes, W, b, activation_func)
        


# Momentum-based Gradient Descent optimizer.

def momentum_gd(x_train, y_train, hiddenlayers, hiddennodes, input_nodes, output_nodes,
                W, b, a, h, learning_rate, num_epochs, batch_size, train_set,
                activation_func, loss_func, weight_decay, momentum):
    # Initialize velocity terms.
    velocity_weights = {}
    velocity_biases = {}
    for epoch in range(num_epochs):
        gradient_weights = {}
        gradient_biases = {}
        for index in range(train_set):
            yhat, a, h = forward_propagation(x_train, hiddenlayers, hiddennodes, input_nodes,
                                             output_nodes, W, b, a, h, index, activation_func)
            back_propagation(y_train, hiddenlayers, hiddennodes, input_nodes, output_nodes,
                             W, b, a, h, gradient_weights, gradient_biases, index, yhat, activation_func, loss_func)
            if index % batch_size == 0:
                i = 1
                while i < hiddenlayers + 2:
                    if i not in velocity_weights:
                        velocity_weights[i] = np.zeros_like(W[i])
                        velocity_biases[i] = np.zeros_like(b[i])
                    # Update velocity terms.
                    velocity_weights[i] = momentum * velocity_weights[i] + learning_rate * (gradient_weights[i] + weight_decay * W[i])
                    velocity_biases[i] = momentum * velocity_biases[i] + learning_rate * gradient_biases[i]
                    # Update weights and biases using the velocity.
                    W[i] -= velocity_weights[i]
                    b[i] -= velocity_biases[i]
                    i += 1
                gradient_weights, gradient_biases = {}, {}

        # Log epoch number.
        wandb.log({"epoch " : epoch})
        
        # Evaluate and log performance on test, validation, and training sets.
        evaluate_performance_train(x_train, y_train, hiddenlayers, hiddennodes, input_nodes, output_nodes, W, b, activation_func, loss_func)
        evaluate_performance_validity(x_train, y_train, hiddenlayers, hiddennodes, input_nodes, output_nodes, W, b, activation_func, loss_func)
        evaluate_performance_test(x_test, y_test, hiddenlayers, hiddennodes, input_nodes, output_nodes, W, b, activation_func)



# Nesterov Accelerated Gradient (NAG) optimizer.

def nesterov_gd(x_train, y_train, hiddenlayers, hiddennodes, input_nodes, output_nodes,
                W, b, a, h, learning_rate, num_epochs, batch_size, train_set,
                activation_func, loss_func, weight_decay, momentum):
    velocity_weights = {}
    velocity_biases = {}
    for epoch in range(num_epochs):
        gradient_weights = {}
        gradient_biases = {}
        for index in range(train_set):
            # Create lookahead weights and biases.
            W_temp = {}
            b_temp = {}
            for i in range(1, hiddenlayers + 2):
                if i not in velocity_weights:
                    velocity_weights[i] = np.zeros_like(W[i])
                    velocity_biases[i] = np.zeros_like(b[i])
                W_temp[i] = W[i] - momentum * velocity_weights[i]
                b_temp[i] = b[i] - momentum * velocity_biases[i]
            # Forward pass with lookahead parameters.
            yhat, a_temp, h_temp = forward_propagation(x_train, hiddenlayers, hiddennodes, input_nodes,
                                                       output_nodes, W_temp, b_temp, a, h, index, activation_func)
            back_propagation(y_train, hiddenlayers, hiddennodes, input_nodes, output_nodes,
                             W_temp, b_temp, a_temp, h_temp, gradient_weights, gradient_biases, index, yhat, activation_func, loss_func)
            if index % batch_size == 0:
                i = 1
                while i < hiddenlayers + 2:
                    velocity_weights[i] = momentum * velocity_weights[i] + learning_rate * (gradient_weights[i] + weight_decay * W[i])
                    velocity_biases[i] = momentum * velocity_biases[i] + learning_rate * gradient_biases[i]
                    W[i] -= velocity_weights[i]
                    b[i] -= velocity_biases[i]
                    i += 1
                gradient_weights, gradient_biases = {}, {}
        
        # Log epoch number.
        wandb.log({"epoch " : epoch})
        
        # Evaluate and log performance on test, validation, and training sets.
        evaluate_performance_train(x_train, y_train, hiddenlayers, hiddennodes, input_nodes, output_nodes, W, b, activation_func, loss_func)
        evaluate_performance_validity(x_train, y_train, hiddenlayers, hiddennodes, input_nodes, output_nodes, W, b, activation_func, loss_func)
        evaluate_performance_test(x_test, y_test, hiddenlayers, hiddennodes, input_nodes, output_nodes, W, b, activation_func)


# RMSProp optimizer.

def rmsprop(x_train, y_train, hiddenlayers, hiddennodes, input_nodes, output_nodes,
            W, b, a, h, learning_rate, num_epochs, batch_size, train_set,
            activation_func, loss_func, weight_decay, beta, epsilon):
    # Initialize squared gradient accumulators.
    squared_grad_weights = {}
    squared_grad_biases = {}
    for epoch in range(num_epochs):
        gradient_weights = {}
        gradient_biases = {}
        for index in range(train_set):
            yhat, a_temp, h_temp = forward_propagation(x_train, hiddenlayers, hiddennodes,
                                                       input_nodes, output_nodes, W, b, a, h, index, activation_func)
            back_propagation(y_train, hiddenlayers, hiddennodes, input_nodes, output_nodes,
                             W, b, a, h, gradient_weights, gradient_biases, index, yhat, activation_func, loss_func)
            if index % batch_size == 0:
                i = 1
                while i < hiddenlayers + 2:
                    if i not in squared_grad_weights:
                        squared_grad_weights[i] = np.zeros_like(W[i])
                        squared_grad_biases[i] = np.zeros_like(b[i])
                    grad_W = gradient_weights[i] + weight_decay * W[i]
                    grad_b = gradient_biases[i]
                    # Update running averages of squared gradients.
                    squared_grad_weights[i] = beta * squared_grad_weights[i] + (1 - beta) * np.square(grad_W)
                    squared_grad_biases[i] = beta * squared_grad_biases[i] + (1 - beta) * np.square(grad_b)
                    # Update weights and biases.
                    W[i] -= learning_rate * grad_W / (np.sqrt(squared_grad_weights[i]) + epsilon)
                    b[i] -= learning_rate * grad_b / (np.sqrt(squared_grad_biases[i]) + epsilon)
                    i += 1
                gradient_weights, gradient_biases = {}, {}

        # Log epoch number.
        wandb.log({"epoch " : epoch})
        
        # Evaluate and log performance on test, validation, and training sets.
        evaluate_performance_train(x_train, y_train, hiddenlayers, hiddennodes, input_nodes, output_nodes, W, b, activation_func, loss_func)
        evaluate_performance_validity(x_train, y_train, hiddenlayers, hiddennodes, input_nodes, output_nodes, W, b, activation_func, loss_func)
        evaluate_performance_test(x_test, y_test, hiddenlayers, hiddennodes, input_nodes, output_nodes, W, b, activation_func)


# Adam optimizer.

def adam(x_train, y_train, hiddenlayers, hiddennodes, input_nodes, output_nodes,
         W, b, a, h, learning_rate, num_epochs, batch_size, train_set,
         activation_func, loss_func, weight_decay, beta1, beta2, epsilon):
    # Initialize first moment (m) and second moment (v) estimates.
    first_momentum_weights = {}
    second_momentum_weights = {}
    first_momentum_biases = {}
    second_momentum_biases = {}
    t = 0  # Time step counter.
    for epoch in range(num_epochs):
        gradient_weights = {}
        gradient_biases = {}
        for index in range(train_set):
            yhat, a_temp, h_temp = forward_propagation(x_train, hiddenlayers, hiddennodes,
                                                       input_nodes, output_nodes, W, b, a, h, index, activation_func)
            back_propagation(y_train, hiddenlayers, hiddennodes, input_nodes, output_nodes,
                             W, b, a, h, gradient_weights, gradient_biases, index, yhat, activation_func, loss_func)
            if index % batch_size == 0:
                t += 1  # Increment time step.
                i = 1
                while i < hiddenlayers + 2:
                    if i not in first_momentum_weights:
                        first_momentum_weights[i] = np.zeros_like(W[i])
                        second_momentum_weights[i] = np.zeros_like(W[i])
                        first_momentum_biases[i] = np.zeros_like(b[i])
                        second_momentum_biases[i] = np.zeros_like(b[i])
                    grad_W = gradient_weights[i] + weight_decay * W[i]
                    grad_b = gradient_biases[i]
                    # Update biased first moment estimate.
                    first_momentum_weights[i] = beta1 * first_momentum_weights[i] + (1 - beta1) * grad_W
                    first_momentum_biases[i] = beta1 * first_momentum_biases[i] + (1 - beta1) * grad_b
                    # Update biased second raw moment estimate.
                    second_momentum_weights[i] = beta2 * second_momentum_weights[i] + (1 - beta2) * np.square(grad_W)
                    second_momentum_biases[i] = beta2 * second_momentum_biases[i] + (1 - beta2) * np.square(grad_b)
                    # Compute bias-corrected first moment estimates.
                    mW_hat = first_momentum_weights[i] / (1 - beta1**t)
                    mB_hat = first_momentum_biases[i] / (1 - beta1**t)
                    # Compute bias-corrected second moment estimates.
                    vW_hat = second_momentum_weights[i] / (1 - beta2**t)
                    vB_hat = second_momentum_biases[i] / (1 - beta2**t)
                    # Update weights and biases.
                    W[i] -= learning_rate * mW_hat / (np.sqrt(vW_hat) + epsilon)
                    b[i] -= learning_rate * mB_hat / (np.sqrt(vB_hat) + epsilon)
                    i += 1
                gradient_weights, gradient_biases = {}, {}

        # Log epoch number.
        wandb.log({"epoch " : epoch})
        
        # Evaluate and log performance on test, validation, and training sets.
        evaluate_performance_train(x_train, y_train, hiddenlayers, hiddennodes, input_nodes, output_nodes, W, b, activation_func, loss_func)
        evaluate_performance_validity(x_train, y_train, hiddenlayers, hiddennodes, input_nodes, output_nodes, W, b, activation_func, loss_func)
        evaluate_performance_test(x_test, y_test, hiddenlayers, hiddennodes, input_nodes, output_nodes, W, b, activation_func)


# Nadam optimizer.

def nadam(x_train, y_train, hiddenlayers, hiddennodes, input_nodes, output_nodes,
          W, b, a, h, learning_rate, num_epochs, batch_size, train_set,
          activation_func, loss_func, weight_decay, beta1, beta2, epsilon):
    first_momentum_weights = {}
    second_momentum_weights = {}
    first_momentum_biases = {}
    second_momentum_biases = {}
    t = 0  # Time step counter.
    for epoch in range(num_epochs):
        gradient_weights = {}
        gradient_biases = {}
        for index in range(train_set):
            yhat, a_temp, h_temp = forward_propagation(x_train, hiddenlayers, hiddennodes,
                                                       input_nodes, output_nodes, W, b, a, h, index, activation_func)
            back_propagation(y_train, hiddenlayers, hiddennodes, input_nodes, output_nodes,
                             W, b, a, h, gradient_weights, gradient_biases, index, yhat, activation_func, loss_func)
            if index % batch_size == 0:
                t += 1  # Increment time step.
                i = 1
                while i < hiddenlayers + 2:
                    if i not in first_momentum_weights:
                        first_momentum_weights[i] = np.zeros_like(W[i])
                        second_momentum_weights[i] = np.zeros_like(W[i])
                        first_momentum_biases[i] = np.zeros_like(b[i])
                        second_momentum_biases[i] = np.zeros_like(b[i])
                    grad_W = gradient_weights[i] + weight_decay * W[i]
                    grad_b = gradient_biases[i]
                    # Update biased first moment estimate.
                    first_momentum_weights[i] = beta1 * first_momentum_weights[i] + (1 - beta1) * grad_W
                    first_momentum_biases[i] = beta1 * first_momentum_biases[i] + (1 - beta1) * grad_b
                    # Update biased second raw moment estimate.
                    second_momentum_weights[i] = beta2 * second_momentum_weights[i] + (1 - beta2) * np.square(grad_W)
                    second_momentum_biases[i] = beta2 * second_momentum_biases[i] + (1 - beta2) * np.square(grad_b)
                    # Compute bias-corrected estimates.
                    mW_hat = first_momentum_weights[i] / (1 - beta1**t)
                    mB_hat = first_momentum_biases[i] / (1 - beta1**t)
                    vW_hat = second_momentum_weights[i] / (1 - beta2**t)
                    vB_hat = second_momentum_biases[i] / (1 - beta2**t)
                    # Nadam update rule for weights and biases.
                    nadam_update_W = (beta1 * mW_hat) + ((1 - beta1) * grad_W) / (1 - beta1**t)
                    nadam_update_B = (beta1 * mB_hat) + ((1 - beta1) * grad_b) / (1 - beta1**t)
                    W[i] -= learning_rate * nadam_update_W / (np.sqrt(vW_hat) + epsilon)
                    b[i] -= learning_rate * nadam_update_B / (np.sqrt(vB_hat) + epsilon)
                    i += 1
                gradient_weights, gradient_biases = {}, {}
                
        # Log epoch number.
        wandb.log({"epoch " : epoch})
        
        # Evaluate and log performance on test, validation, and training sets.
        evaluate_performance_train(x_train, y_train, hiddenlayers, hiddennodes, input_nodes, output_nodes, W, b, activation_func, loss_func)
        evaluate_performance_validity(x_train, y_train, hiddenlayers, hiddennodes, input_nodes, output_nodes, W, b, activation_func, loss_func)
        evaluate_performance_test(x_test, y_test, hiddenlayers, hiddennodes, input_nodes, output_nodes, W, b, activation_func)


##########################
# Training Function      #
##########################

def train_model(x_train, y_train, hidden_layers, hidden_nodes, input_nodes, output_nodes,
                W, b, a, h, lr, epochs, batch, n_train, opt, act_func, loss_func, init_method,
                weight_decay, momentum, beta1, beta2, epsilon, beta):
    # Initialize weights and biases using the specified method.
    init(W, b, input_nodes, hidden_layers, hidden_nodes, output_nodes, init_method)
    
    # Choose the optimizer based on user input.
    if opt == "nadam":
        nadam(x_train, y_train, hidden_layers, hidden_nodes, input_nodes, output_nodes,
              W, b, a, h, lr, epochs, batch, n_train, act_func, loss_func, weight_decay, beta1, beta2, epsilon)
    elif opt == "adam":
        adam(x_train, y_train, hidden_layers, hidden_nodes, input_nodes, output_nodes,
             W, b, a, h, lr, epochs, batch, n_train, act_func, loss_func, weight_decay, beta1, beta2, epsilon)
    elif opt == "rmsprop":
        rmsprop(x_train, y_train, hidden_layers, hidden_nodes, input_nodes, output_nodes,
                W, b, a, h, lr, epochs, batch, n_train, act_func, loss_func, weight_decay, beta, epsilon)
    elif opt == "nag":
        nesterov_gd(x_train, y_train, hidden_layers, hidden_nodes, input_nodes, output_nodes,
                     W, b, a, h, lr, epochs, batch, n_train, act_func, loss_func, weight_decay, momentum)
    elif opt == "momentum":
        momentum_gd(x_train, y_train, hidden_layers, hidden_nodes, input_nodes, output_nodes,
                    W, b, a, h, lr, epochs, batch, n_train, act_func, loss_func, weight_decay, momentum)
    elif opt == "sgd":
        SGD(x_train, y_train, hidden_layers, hidden_nodes, input_nodes, output_nodes,
            W, b, a, h, lr, epochs, batch, n_train, act_func, loss_func, weight_decay)
    else:
        raise ValueError("Optimizer option not recognized.")

    # Evaluate and return the final performance after training.
    train_accuracy, train_loss = evaluate_performance_train(x_train, y_train, hidden_layers, hidden_nodes, input_nodes, output_nodes, W, b, act_func, loss_func)
    val_accuracy, val_loss = evaluate_performance_validity(x_train, y_train, hidden_layers, hidden_nodes, input_nodes, output_nodes, W, b, act_func, loss_func)
    test_accuracy = evaluate_performance_test(x_test, y_test, hidden_layers, hidden_nodes, input_nodes, output_nodes, W, b, act_func)
    return train_accuracy, val_accuracy, test_accuracy



##########################
# Main Function          #
##########################

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description='Train a neural network on MNIST or Fashion MNIST.')
    parser.add_argument('-wp', '--wandb_project', type=str, default='DA6401-Assignment1',
                        help='WandB project name')
    parser.add_argument('-we', '--wandb_entity', type=str, default='your_entity',
                        help='WandB entity name')
    parser.add_argument('-d', '--dataset', type=str, choices=['mnist', 'fashion_mnist'],
                        default='fashion_mnist', help='Dataset selection')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('-l', '--loss', type=str,
                        choices=['cross_entropy', 'mean_squared_error'],
                        default='cross_entropy', help='Loss function')
    parser.add_argument('-o', '--optimizer', type=str,
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],
                        default='adam', help='Optimizer')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('-w_i', '--weight_init', type=str,
                        choices=['random', 'Xavier'], default='Xavier',
                        help='Weight initialization method')
    parser.add_argument('-nhl', '--num_layers', type=int, default=5,
                        help='Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', type=int, default=64,
                        help='Size of each hidden layer')
    parser.add_argument('-a', '--activation', type=str,
                        choices=['sigmoid', 'tanh', 'relu'], default='relu',
                        help='Activation function')
    parser.add_argument('-m', '--momentum', type=float, default=0.9,
                        help='Momentum value for momentum-based optimizers')
    parser.add_argument('-beta1', '--beta1', type=float, default=0.9,
                        help='Beta1 value for Adam/Nadam')
    parser.add_argument('-beta2', '--beta2', type=float, default=0.999,
                        help='Beta2 value for Adam/Nadam')
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-8,
                        help='Epsilon value for optimizers')
    parser.add_argument('-beta', '--beta', type=float, default=0.9,
                        help='Decay rate (beta) for RMSprop')
    
    args = parser.parse_args()
    
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    print("Training started with the following configurations:")
    print(args)

    global x_test, y_test
    
    if args.dataset == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.reshape(x_train.shape[0], 784) / 256.0
    x_test = x_test.reshape(x_test.shape[0], 784) / 256.0
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)
    
    input_size = 784  # Input nodes for 28x28 images.
    num_classes = 10  # Number of output classes.
    if x_train.shape[0] >= 60000:
        n_train_samples = 54000  # Use 54000 samples for training if dataset is large.
    else:
        n_train_samples = int(x_train.shape[0] * 0.9)  # Else, use 90% of available samples.
    
    loss_mode = args.loss
    weights = {}
    biases = {}
    a = {}
    h = {}
    
    final_train_acc, final_val_acc, final_test_acc = train_model(
        x_train, y_train,
        args.num_layers, args.hidden_size,
        input_size, num_classes,
        weights, biases, a, h,
        args.learning_rate, args.epochs, args.batch_size,
        n_train_samples, args.optimizer,
        args.activation, loss_mode, args.weight_init, args.weight_decay,
        args.momentum, args.beta1, args.beta2, args.epsilon, args.beta
    )
    
    print("Final Training Accuracy: {:.2f}%".format(final_train_acc))
    print("Final Validation Accuracy: {:.2f}%".format(final_val_acc))
    print("Final Test Accuracy: {:.2f}%".format(final_test_acc))
    wandb.finish()

if __name__ == '__main__':
    main()

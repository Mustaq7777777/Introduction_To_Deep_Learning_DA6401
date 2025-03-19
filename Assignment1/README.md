## Neural Network Training Script

This repository contains a Python script (`train.py`) that implements a configurable feedforward neural network for classifying images from the MNIST and Fashion MNIST datasets. The network supports multiple optimizers (SGD, Momentum, NAG, RMSProp, Adam, Nadam) and various hyperparameter settings. All training progress and performance metrics are logged to Weights & Biases (W&B), allowing for detailed experiment tracking and visualization.

## WandB Report

Check out the detailed experiment report on [WandB](https://api.wandb.ai/links/cs24m045-indian-institute-of-technology-madras/dhrf1snl)

## Repository

The complete code and documentation for this project can be found on the [GitHub Repository](https://github.com/Mustaq7777777/Introduction_To_Deep_Learning_DA6401/tree/main/Assignment1).


## Table of Contents
- Overview
- Features
- Installation
- Usage
- Supported Command-Line Arguments
- Project Structure
- Experiment Tracking with W&B
- Example Commands

## Overview
The `train.py` script trains a feedforward neural network on either the MNIST or Fashion MNIST dataset. It supports multiple optimizers and allows you to customize many aspects of the training process through command-line arguments. All metrics—including training, validation, and test performance—are logged to W&B for real-time monitoring and comparison of experiments.

## Features
- **Multiple Optimizers:** Supports SGD, Momentum, NAG, RMSProp, Adam, and Nadam.
- **Configurable Architecture:** Set the number of hidden layers and the number of neurons per layer.
- **Customizable Hyperparameters:** Tune learning rate, batch size, weight decay, momentum, and other parameters.
- **Activation Functions:** Choose from sigmoid, tanh, or ReLU.
- **Weight Initialization:** Options include random and Xavier initialization.
- **Detailed Evaluation:** Separate evaluation functions for training, validation, and test performance.
- **Experiment Logging:** Integrated with Weights & Biases for comprehensive experiment tracking.

## Installation
### Clone the Repository:
```bash
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
```
### Install Dependencies:
Ensure you have Python 3.7+ installed, then run:
```bash
pip install wandb numpy matplotlib keras
```

## Usage
Run the script from the command line, providing the desired parameters. For example:
```bash
python train.py -wp "DA6401-Assignment1" -we "your_entity" -d fashion_mnist -e 10 -b 64 -l cross_entropy -o adam -lr 0.001 -w_d 0.0 -w_i Xavier -nhl 5 -sz 64 -a relu -m 0.9 -beta1 0.9 -beta2 0.999 -eps 1e-8 -beta 0.9
```

## Supported Command-Line Arguments
| Short | Long | Default | Description | Possible Values |
|-------|------|---------|-------------|-----------------|
| `-wp` | `--wandb_project` | DA6401-Assignment1 | Name of the W&B project. | Any string |
| `-we` | `--wandb_entity` | your_entity | W&B entity (user or team name). | Any string |
| `-d` | `--dataset` | fashion_mnist | Dataset to train on. | mnist or fashion_mnist |
| `-e` | `--epochs` | 10 | Number of training epochs. | Any positive integer |
| `-b` | `--batch_size` | 64 | Batch size used during training. | Any positive integer |
| `-l` | `--loss` | cross_entropy | Loss function to use. | cross_entropy or mean_squared_error |
| `-o` | `--optimizer` | adam | Optimizer used for training. | sgd, momentum, nag, rmsprop, adam, or nadam |
| `-lr` | `--learning_rate` | 0.001 | Learning rate for the optimizer. | Any positive float |
| `-w_d` | `--weight_decay` | 0.0 | Weight decay (L2 regularization). | Any non-negative float |
| `-w_i` | `--weight_init` | Xavier | Weight initialization method. | random or Xavier |
| `-nhl` | `--num_layers` | 5 | Number of hidden layers in the network. | Any positive integer |
| `-sz` | `--hidden_size` | 64 | Number of neurons per hidden layer. | Any positive integer |
| `-a` | `--activation` | relu | Activation function used in the network. | sigmoid, tanh, or relu |
| `-m` | `--momentum` | 0.9 | Momentum coefficient for momentum-based optimizers. | Float between 0 and 1 |
| `-beta1` | `--beta1` | 0.9 | Beta1 parameter for Adam/Nadam optimizers. | Float between 0 and 1 |
| `-beta2` | `--beta2` | 0.999 | Beta2 parameter for Adam/Nadam optimizers. | Float between 0 and 1 |
| `-eps` | `--epsilon` | 1e-8 | Small constant for numerical stability in optimizers. | Any small positive float |
| `-beta` | `--beta` | 0.9 | Decay rate for RMSProp optimizer. | Float between 0 and 1 |

## Project Structure
- **Activation Functions:** Implements sigmoid, tanh, and ReLU along with their derivatives.
- **Weight Initialization:** Provides both Xavier and random initialization methods.
- **Forward/Backward Propagation:** Uses while loops for propagating activations and errors through the network.
- **Optimizers:** Contains separate functions for SGD, Momentum, NAG, RMSProp, Adam, and Nadam with batch updates.
- **Evaluation:** Divided into three functions to evaluate training, validation, and test performance.
- **Main Function:** Parses command-line arguments, loads data, and initiates the training process.

## Experiment Tracking with W&B
All training metrics are logged to Weights & Biases:
- **Training Metrics:** Training accuracy and loss are logged after each epoch.
- **Validation Metrics:** Validation accuracy and loss (evaluated on the last 6000 training samples) are logged.
- **Test Metrics:** Test accuracy is logged after each epoch.
These logs allow you to monitor progress and compare different experiments in real time via the W&B dashboard.

## Example Commands
### Fashion MNIST, Adam optimizer, Xavier initialization:
```bash
python train.py -wp "DA6401-Assignment1" -we "your_entity" -d fashion_mnist -e 10 -b 64 -l cross_entropy -o adam -lr 0.001 -w_d 0.0 -w_i Xavier -nhl 5 -sz 64 -a relu -m 0.9 -beta1 0.9 -beta2 0.999 -eps 1e-8 -beta 0.9
```
### MNIST, Momentum optimizer, random initialization:
```bash
python train.py -wp "DA6401-Assignment1" -we "your_entity" -d mnist -e 10 -b 64 -l mean_squared_error -o momentum -lr 0.001 -w_d 0.0 -w_i random -nhl 5 -sz 64 -a relu -m 0.9 -beta1 0.9 -beta2 0.999 -eps 1e-8 -beta 0.9
```
## Note
Use "ReLU" instead of "relu" when using commands as specified in the report

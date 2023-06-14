import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))
def derivative_s(x):
    return x*(1-x)
def relu(x):
    return np.maximum(x, 0)
def derivative_r(x):
    return np.where(x > 0, 1, 0)
def tanh(x):
    return (np.exp(x)- np.exp(-x))/ (np.exp(x) + np.exp(-x))
def derivative_t(x):
    return 1 - np.tanh(x)**2
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha*x)
def derivative_lr(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)
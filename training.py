import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from discriminator import Discriminator
from generator import Generator

print("Loading data...")

data3 = pd.read_csv("creditcard.csv")
data3['normAmount'] = StandardScaler().fit_transform(data3['Amount'].values.reshape(-1, 1))
data3 = data3.drop(['Time','Amount'],axis=1)

X = np.array(data3.ix[:, data3.columns != 'Class'])
y = np.array(data3.ix[:, data3.columns == 'Class'])

from torch.autograd import Variable
X = Variable(torch.FloatTensor(X))
y = Variable(torch.FloatTensor(y))

print("Starting generator and discriminator...")

# Generator's paragrams
g_input_size = 29     # Random noise dimension
g_hidden_size = 50   # Generator complexity
g_output_size = 1
g_learning_rate = 0.0002
g_steps = 1

#Discriminator's paragrams
d_input_size = 29   # Minibatch size
d_hidden_size = 50  # Discriminator complexity
d_output_size = 1   # Single dimension for 'real' vs. 'fake'
d_learning_rate = 0.0002
d_steps = 1

minibatch_size = d_input_size

num_epochs = 1000
print_interval = 10

generator = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
discriminator = Discriminator(input_size=d_input_func(d_input_size), hidden_size=d_hidden_size, output_size=d_output_size)

def extract(v):
    return v.data.storage().tolist()

def stats(d):
    return [np.mean(d), np.std(d)]

# Use Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Set the optimizers
beta_1 = 0.5
beta_2 = 0.999
d_optimizer = optim.Adam(discriminator.parameters(), lr=d_learning_rate/2, betas=(beta_1, beta_2))
g_optimizer = optim.Adam(generator.parameters(), lr=g_learning_rate, betas=(beta_1, beta_2))

# Training GANs
for epoch in range(num_epochs):
    for d_index in range(d_steps):
        # 1. Updating the weights of the Discriminator
        discriminator.zero_grad()  # Initialize gradients of the Discriminator to 0

        # Training the Discriminator with real data
        d_real_data = Variable(X[0])  # Wrap it in a variable
        d_real_decision = discriminator(d_real_data)  # Forward propagate this real data into the neural network
        # target = Variable(torch.ones(input.size()[0]))
        y_real = Variable(torch.ones(1))  # Get the target
        d_real_loss = BCE_loss(d_real_decision, y_real)  # Compute the loss between the prediction and actual
        d_real_loss.backward()  # Compute/store gradients

        # Train the Discriminator with a fake data generated by the Generator
        # d_gen_input = Variable(torch.randn(minibatch_size, 29))
        d_gen_input = Variable(torch.randn(minibatch_size, g_input_size))
        d_fake_data = generator(d_gen_input).detach()  # detach to avoid training G on these labels

        d_fake_decision = discriminator(d_fake_data.t())
        y_fake = Variable(torch.zeros(1))
        d_fake_loss = BCE_loss(d_fake_decision, y_fake)  # zeros = fake
        d_fake_loss.backward()
        d_optimizer.step()  # Apply SGD to update the weight

    for g_index in range(g_steps):
        # 2. Update the weight of the Generator
        generator.zero_grad()

        gen_input = Variable(torch.randn(minibatch_size, g_input_size))
        g_fake_data = generator(gen_input)
        dg_fake_decision = discriminator(g_fake_data.t())
        target = Variable(torch.ones(1))
        g_loss = BCE_loss(dg_fake_decision, target)
        g_loss.backward()
        g_optimizer.step()  # Only optimizes G's parameters

    if epoch % print_interval == 0:
        print("%s: Discriminator: Real Loss %s / Fake Loss %s Generator: %s (Real Data: %s, Fake Data: %s) " % (epoch,
                                                                                                                extract(d_real_loss)[0],
                                                                                                                extract(d_fake_loss)[0],
                                                                                                                extract(g_loss)[0],
                                                                                                                stats(extract(d_real_data)),
                                                                                                                stats(extract(d_fake_data))))

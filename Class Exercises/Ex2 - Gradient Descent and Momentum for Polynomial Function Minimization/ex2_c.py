# -*- coding: utf-8 -*-

# square of x
def x2(x):
    return (x * x)


# derivative of x2
def x2_(x):
    # your code here:
    return 2 * x


# starting point
X = 10

# your code here:
lr = 0.01
num_of_steps = 1000
for i in range(num_of_steps):
    # your code here:
    X -= lr * x2_(X)
"""## Ex0 (b) find the minimums of x^4 using Gradient Descent"""

print(f'SGD_X^2: {X}')


# x to the power of 4
def x4(x):
    return (x * x * x * x)


# derivative of x4
def x4_(x):
    # your code here:
    return 4 * (x * x * x)


# starting point
X = 10

# your code here:
lr = 0.001
num_of_steps = 50000
for i in range(num_of_steps):
    X -= lr * x4_(X)
print(f'SGD_X^4: {X}')
"""## Ex1 - find the minimums of x^2 and x^4 using the Momentum methos and compare it to Gradient Descent"""

# starting point for the Gradient Descent
X2 = 10
X4 = 10

# your code here:
lr = 0.001

# your code here:
num_of_steps = 50000
for i in range(num_of_steps):
    # your code here:
    X2 -= lr * x2_(X2)
    X4 -= lr * x4_(X4)
print("X2:{} \t X4:{} ".format(X2, X4))

# starting point for the Momentum methos
X2m = 10
X4m = 10

# your code here (Find the appropriate learning rate, one that works for both functions)
lrm = 0.001
mu = 0.9
vx2 = 0
vx4 = 0
# your code here:
num_of_steps = 10000
for i in range(num_of_steps):
    vx2 = mu * vx2 - lrm * (x2_(X2m))
    X2m += vx2
    vx4 = mu * vx4 - lrm * (x4_(X4m))
    X4m += vx4
print("X2m:{} \t X4m:{}".format(X2m, X4m))

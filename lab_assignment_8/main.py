import numpy as np
import math
import matplotlib.pyplot as plt

def summation(x,w):
    s=0
    for i in range(len(x)):
        s=s+(x[i]*w[i])
    return s

def step(x):
    if x>=0:
        return 1
    else:
        return 0

def bipolar_step(x):
    if x>=0:
        return 1
    else:
        return -1

def sigmoid(x):
    value=1/(1+math.exp(-x))
    return value

def tanh(x):
    return math.tanh(x)

def relu(x):
    if x>0:
        return x
    else:
        return 0

def leaky_relu(x):
    if x>0:
        return x
    else:
        return 0.01*x

def comparator(o,t):
    error=t-o
    return error

print("A1\n")
x=[2,1]
w=[0.5,0.3]

s=summation(x,w)

print("summation :",s)
print("step :",step(s))
print("bipolar step :",bipolar_step(s))
print("sigmoid :",sigmoid(s))
print("tanH :",tanh(s))
print("reLU :",relu(s))
print("leaky relu :",leaky_relu(s))
print("error :",comparator(1,0))
print("\n")

print("A2\n")
def step(net):
    if net>=0:
        return 1
    else:
        return 0

inputs=[[0,0],[0,1],[1,0],[1,1]]
targets=[0,0,0,1]
w0=10
w1=0.2
w2=-0.75
lr=0.05
epoch=0
sse_list=[]
while epoch<1000:
    sse=0
    for i in range(len(inputs)):
        net=w0+(inputs[i][0]*w1)+(inputs[i][1]*w2)
        output=step(net)
        error=targets[i]-output
        sse=sse+(error*error)
        w0=w0+(lr*error)
        w1=w1+(lr*error*inputs[i][0])
        w2=w2+(lr*error*inputs[i][1])
    sse_list.append(sse)
    epoch=epoch+1
    if sse<=0.002:
        break

print("epochs :",epoch)
print("w0 :",w0)
print("w1 :",w1)
print("w2 :",w2)

plt.plot(range(1,epoch+1),sse_list)
plt.xlabel("epoch")
plt.ylabel("sse")
plt.show()
print("\n")

print("A3\n")
def train(act_func,targets):
    inputs=[[0,0],[0,1],[1,0],[1,1]]
    w0=10
    w1=0.2
    w2=-0.75
    lr=0.05
    epoch=0
    while epoch<1000:
        sse=0
        for i in range(len(inputs)):
            net=w0+(inputs[i][0]*w1)+(inputs[i][1]*w2)
            output=act_func(net)
            error=targets[i]-output
            sse=sse+(error*error)
            w0=w0+(lr*error)
            w1=w1+(lr*error*inputs[i][0])
            w2=w2+(lr*error*inputs[i][1])
        epoch=epoch+1
        if sse<=0.002:
            break
    return epoch

bipolar_epochs=train(bipolar_step,[-1,-1,-1,1])
sigmoid_epochs=train(sigmoid,[0,0,0,1])
relu_epochs=train(relu,[0,0,0,1])

print("bipolar step epochs :",bipolar_epochs)
print("sigmoid epochs :",sigmoid_epochs)
print("relu epochs :",relu_epochs)
print("\n")


print("A4\n")
def train_lr(lr):
    inputs=[[0,0],[0,1],[1,0],[1,1]]
    targets=[0,0,0,1]
    w0=10
    w1=0.2
    w2=-0.75
    epoch=0
    while epoch<1000:
        sse=0
        for i in range(len(inputs)):
            net=w0+(inputs[i][0]*w1)+(inputs[i][1]*w2)
            output=step(net)
            error=targets[i]-output
            sse=sse+(error*error)
            w0=w0+(lr*error)
            w1=w1+(lr*error*inputs[i][0])
            w2=w2+(lr*error*inputs[i][1])
        epoch=epoch+1
        if sse<=0.002:
            break
    return epoch

lr_list=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
epoch_list=[]
for lr in lr_list:
    epochs=train_lr(lr)
    epoch_list.append(epochs)
    print("learning rate :",lr)
    print("epochs :",epochs)
plt.plot(lr_list,epoch_list,marker="o")
plt.xlabel("learning rate")
plt.ylabel("epochs")
plt.show()
print("\n")

print("A5 using XOR logic\n")
inputs=[[0,0],[0,1],[1,0],[1,1]]
targets=[0,1,1,0]
w0=10
w1=0.2
w2=-0.75
lr=0.05
epoch=0
while epoch<1000:
    sse=0
    for i in range(len(inputs)):
        net=w0+(inputs[i][0]*w1)+(inputs[i][1]*w2)
        output=step(net)
        error=targets[i]-output
        sse=sse+(error*error)
        w0=w0+(lr*error)
        w1=w1+(lr*error*inputs[i][0])
        w2=w2+(lr*error*inputs[i][1])
    epoch=epoch+1
    if sse<=0.002:
        break
print("epochs :",epoch)
print("sse :",sse)
print("\n")

print("A5 activation comparison using XOR logic\n")
bipolar_epochs=train(bipolar_step,[-1,1,1,-1])
sigmoid_epochs=train(sigmoid,[0,1,1,0])
relu_epochs=train(relu,[0,1,1,0])
print("bipolar step epochs :",bipolar_epochs)
print("sigmoid epochs :",sigmoid_epochs)
print("relu epochs :",relu_epochs)
print("\n")


print("a6\n")
inputs=[[20,6,2,386],[16,3,6,289],[27,6,2,393],[19,1,2,110],[24,4,2,280],[22,1,5,167],[15,4,2,271],[18,4,2,274],[21,1,4,148],[16,2,4,198]]
targets=[1,1,1,0,1,0,1,1,0,0]
w0=0.5
w1=0.2
w2=0.3
w3=0.1
w4=0.4
lr=0.01
epoch=0
while epoch<1000:
    sse=0
    for i in range(len(inputs)):
        net=w0
        net=net+(inputs[i][0]*w1)
        net=net+(inputs[i][1]*w2)
        net=net+(inputs[i][2]*w3)
        net=net+(inputs[i][3]*w4)
        output=sigmoid(net)
        error=targets[i]-output
        sse=sse+(error*error)
        w0=w0+(lr*error)
        w1=w1+(lr*error*inputs[i][0])
        w2=w2+(lr*error*inputs[i][1])
        w3=w3+(lr*error*inputs[i][2])
        w4=w4+(lr*error*inputs[i][3])
    epoch=epoch+1
    if sse<=0.002:
        break
print("epochs :",epoch)
print("w0 :",w0)
print("w1 :",w1)
print("w2 :",w2)
print("w3 :",w3)
print("w4 :",w4)
print("sse:",sse)
print("\n")


import numpy as np

print("a7\n")
data=[[20,6,2,386],[16,3,6,289],[27,6,2,393],[19,1,2,110],[24,4,2,280],[22,1,5,167],[15,4,2,271],[18,4,2,274],[21,1,4,148],[16,2,4,198]]
target=[1,1,1,0,1,0,1,1,0,0]
data=np.array(data)
target=np.array(target)
bias=np.ones((len(data),1))
data=np.column_stack((bias,data))
pinv=np.linalg.pinv(data)
w=np.dot(pinv,target)
print("weights :",w)
output=np.dot(data,w)
print("outputs is:",output)
print("\n")



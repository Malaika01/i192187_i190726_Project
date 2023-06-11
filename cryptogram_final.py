#!/usr/bin/env python
# coding: utf-8


# import necessary packages
import torch
import torch.nn as nn
import medmnist
from medmnist import INFO

# Define the CNN class that inherits from the nn.Module class
class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()

        # Define the first convolutional layer with 16 filters of size 3x3
        # followed by a batch normalization layer and a ReLU activation function
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        # Define the second convolutional layer with 16 filters of size 3x3
        # followed by a batch normalization layer, a ReLU activation function, and a max pooling layer with kernel size 2x2 and stride 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Define the third convolutional layer with 64 filters of size 3x3
        # followed by a batch normalization layer and a ReLU activation function
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        # Define the fourth convolutional layer with 64 filters of size 3x3
        # followed by a batch normalization layer and a ReLU activation function
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        # Define the fifth convolutional layer with 64 filters of size 3x3 and padding of 1
        # followed by a batch normalization layer, a ReLU activation function, and a max pooling layer with kernel size 2x2 and stride 2
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Define the fully connected layers with 128 units each and ReLU activation functions
        # followed by a linear layer with num_classes units
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    # Define the forward pass method for the CNN
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
# print the version of the medmnist package
print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")
# set the data flag and download flag
data_flag = 'chestmnist'
download = True

# get the information of the current dataset
info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

# Instantiate the CNN model with the specified number of input channels and output classes
model = CNN(in_channels=n_channels, num_classes=n_classes)
m1 = torch.load('model1.pth')
m2 = torch.load('model2.pth')
m3 = torch.load('model3.pth')


m1


w1 = []
w2 = []
w3 = []
#Iterate over all keys in the state dictionary
for key in m1:
    tensor = m1[key]
    # Iterate over all elements of the tensor
    for i in range(tensor.numel()):
        w1.append(int((tensor.view(-1)[i]+1)*100000))
        
#Iterate over all keys in the state dictionary
for key in m2:
    tensor = m2[key]
    # Iterate over all elements of the tensor
    for i in range(tensor.numel()):
        w2.append(int((tensor.view(-1)[i]+1)* 100000))
# Iterate over all keys in the state dictionary
for key in m3:
    tensor = m3[key]
    # Iterate over all elements of the tensor
    for i in range(tensor.numel()):
        w3.append(int((tensor.view(-1)[i]+1)* 100000)) 
        #w3.append(int(tensor.view(-1)[i] ))  
    



w1



from dataclasses import dataclass
import random

inf = float("inf")

@dataclass
class PrimeGaloisField:
    prime: int

    def __contains__(self, field_value: "FieldElement") -> bool:
        # called whenever you do: <FieldElement> in <PrimeGaloisField>
        return 0 <= field_value.value < self.prime
    
@dataclass
class FieldElement:
    value: int
    field: PrimeGaloisField

    def __repr__(self):
        return "0x" + f"{self.value:x}".zfill(64)
        
    @property
    def P(self) -> int:
        return self.field.prime
    
    def __add__(self, other: "FieldElement") -> "FieldElement":
        return FieldElement(
            value=(self.value + other.value) % self.P,
            field=self.field
        )
    
    def __sub__(self, other: "FieldElement") -> "FieldElement":
        return FieldElement(
            value=(self.value - other.value) % self.P,
            field=self.field
        )
    def __rmul__(self, scalar: int) -> "FieldValue":
        return FieldElement(
            value=(abs(self.value) * scalar) % self.P,
            field=self.field
        )

    def __mul__(self, other: "FieldElement") -> "FieldElement":
        return FieldElement(
            value=(self.value * other.value) % self.P,
            field=self.field
        )
        
    def __pow__(self, exponent: int) -> "FieldElement":
        return FieldElement(
            value=pow(self.value, exponent, self.P),
            field=self.field
        )

    def __truediv__(self, other: "FieldElement") -> "FieldElement":
        other_inv = other ** -1
        return self * other_inv
@dataclass
class EllipticCurve:
    a: int
    b: int

    field: PrimeGaloisField
    
    def __contains__(self, point: "Point") -> bool:
        x, y = point.x, point.y
        return y ** 2 == x ** 3 + self.a * x + self.b

    def __post_init__(self):
        # Encapsulate int parameters in FieldElement
        self.a = FieldElement(self.a, self.field)
        self.b = FieldElement(self.b, self.field)
    
        # Check for membership of curve parameters in the field.
        if self.a not in self.field or self.b not in self.field:
            raise ValueError

@dataclass
class Point:
    x: int
    y: int
    
    curve: EllipticCurve
    
    
    def __post_init__(self):
        # Ignore validation for I
        if self.x is None and self.y is None:
            return

        # Encapsulate int coordinates in FieldElement
        self.x = FieldElement(self.x, self.curve.field)
        self.y = FieldElement(self.y, self.curve.field)

        # Verify if the point satisfies the curve equation
        if self not in self.curve:
            raise ValueError         
    def __add__(self, other):
        #################################################################
        # Point Addition for P₁ or P₂ = I   (identity)                  #
        #                                                               #
        # Formula:                                                      #
        #     P + I = P                                                 #
        #     I + P = P                                                 #
        #################################################################
        if self == I:
            return other

        if other == I:
            return self

        #################################################################
        # Point Addition for X₁ = X₂   (additive inverse)               #
        #                                                               #
        # Formula:                                                      #
        #     P + (-P) = I                                              #
        #     (-P) + P = I                                              #
        #################################################################
        if self.x == other.x and self.y == (-1 * other.y):
            return I

        #################################################################
        # Point Addition for X₁ ≠ X₂   (line with slope)                #
        #                                                               #
        # Formula:                                                      #
        #     S = (Y₂ - Y₁) / (X₂ - X₁)                                 #
        #     X₃ = S² - X₁ - X₂                                         #
        #     Y₃ = S(X₁ - X₃) - Y₁                                      #
        #################################################################
        if self.x != other.x:
            x1, x2 = self.x, other.x
            y1, y2 = self.y, other.y

            s = (y2 - y1) / (x2 - x1)
            x3 = s ** 2 - x1 - x2
            y3 = s * (x1 - x3) - y1

            return self.__class__(
                x=x3.value,
                y=y3.value,
                curve=curve256
            )

        #################################################################
        # Point Addition for P₁ = P₂   (vertical tangent)               #
        #                                                               #
        # Formula:                                                      #
        #     S = ∞                                                     #
        #     (X₃, Y₃) = I                                              #
        #################################################################
        if self == other and self.y == inf:
            return I

        #################################################################
        # Point Addition for P₁ = P₂   (tangent with slope)             #
        #                                                               #
        # Formula:                                                      #
        #     S = (3X₁² + a) / 2Y₁         .. ∂(Y²) = ∂(X² + aX + b)    #
        #     X₃ = S² - 2X₁                                             #
        #     Y₃ = S(X₁ - X₃) - Y₁                                      #
        #################################################################
        if self == other:
            x1, y1, a = self.x, self.y, self.curve.a

            s = (3 * x1 ** 2 + a) / (2 * y1)
            x3 = s ** 2 - 2 * x1
            y3 = s * (x1 - x3) - y1

            return self.__class__(
                x=x3.value,
                y=y3.value,
                curve=curve256
            )
   

    def __rmul__(self, scalar: int) -> "Point":
        # Naive approach:
        #
        # result = I
        # for _ in range(scalar):  # or range(scalar % N)
        #     result = result + self
        # return result
        
        # Optimized approach using binary expansion
        current = self
        result = I
        while scalar:
            if scalar & 1:  # same as scalar % 2
                result = result + current
            current = current + current  # point doubling
            scalar >>= 1  # same as scalar / 2
        return result
#      def __neg__(self):
#         if self == I:
#             return self
#         else:
#             return Point(self.x.value, (-1 * self.y).value, self.curve)
    


# Parameters for the Elliptic Curve being used i.e y² = x³ + 2x + 2
p:int=(0xffffffff00000001000000000000000000000000ffffffffffffffffffffffff)
#p: int=(0x080000000000000000000000000000000000000000000000000000001)
p=17
field = PrimeGaloisField(p)
A:int=(0xffffffff00000001000000000000000000000000fffffffffffffffffffffffc)
B:int=(0x5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b)
A=2
B=2
# A:int=(0x000000000000000000000000000000000000000000000000000000000001)
# B:int=(0x00000000000000000000000000000000000000000000000000000000000c9)
curve256 = EllipticCurve(A,B,field)
I = Point(None,None,curve256)



# G(0,1)
# gx =36863
# gy =30618
gx=5
gy=1
# gx:int=(0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798)
# gy:int=(0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)
# gx:int=(0x6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296)
# gy:int=(0x4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5)
# gx:int=(0x02FE13C0537BBC11ACAA07D793DE4E6D5E5C94EEE8)
# gy:int=(0x0289070FB05D38FF58321F2E800536D538CCDAA3D9)
G = Point(gx,gy,curve256)
# N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
N=0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551
# N:int=(0x040000000000000000000292fe77e70c12a4234c33)
#P(0,-1)
# px:int=(0xdde26a69c791605d878e8479f5d24a11e833bca0a6269698d707affb38d310dd)
# py:int=(0x1ddb215f6a285f2034ff7ac98f7623c477748c3b454343142c72af85bc1ee39e)
# px:int=(0xf068f30fae22d8a2da9c9d29a6d90fe84354d1eec1ccc4880cf130d912d6eb75)
# py:int=(0xfe74f9772f2f6044f5c67e230dda10a43f6520d76e5829240b623c9546f14f4b)
# px=99747
# py=472
px=3
py=1
P = Point(px,py,curve256)
# k = random.randint(1, p-1)
# P = k * G
# print(P)
r1 = random.randint(1, p)
r2 = random.randint(1, p)
r3 = r1 + r2

# nP = Point(px,(p-py)%p,curve256)
#nG = Point(gx,(p-gy)%p,curve256)

nP = Point(px,(p-py),curve256)
nG = Point(gx,(p-gy),curve256)

#print("NP",np)
R1 = r1 * P
R2 = r2 * P 
R3 = r3 * nP
#print("R:",R1,R2,R3)



Z1 = []
Z2 = []
Z3 = []

for i in range(100):
    #print(w1)
    Z1.append(abs(r1 + w1[i]) * G)
    #print(Z1[i])




for i in range(100):
    #print(w2[i])
    Z2.append(abs(r2 + w2[i]) * G)



#nr = -r3 + w3
nr = []
for i in range(100):
    nr.append(-r3+w3[i])
    

#Z3 = -nr * nG
Z3 = []
for i in range(100):
    if nr[i]>0:
        nr[i]=-nr[i]
        Z3.append(-nr[i]* nG)
    else:
        Z3.append(-nr[i]* nG)
  
#Rsum
Rsum = R1 + R2 + R3

Rsum

referencePoint=[]

for i in range(30):
    referencePoint.append(Z1[i] + Z2[i] + Z3[i])
    print("Reference point: \n",referencePoint[i])
    print("\n")
    
basePoint = G
end_time = time.time()



#Cycle of Ps
points = []
points.append(P)
check = False
temp = basePoint
while check == False:
    temp += basePoint
    if(temp == I):
        check = True
    points.append(temp)
print("Length of cycle:",len(points),'\n')



import time
start_time = time.time()
iteration = 0
for r in range(10):
    for i in points:
        if(referencePoint[r] == i):
            num = points.index(i)

        iteration += 1
        
#     weight1=(w1[r]/100000)
#     weight2=(w2[r]/100000)
#     weight3=(w3[r]/100000)
    print(w1[r]+w2[r]+w3[r])
    print("\nReference Point matches the point",num,"in the cycle")
end_time = time.time()
print("Time taken:", end_time - start_time)


import torch
import mlflow
import mlflow.pytorch
import medmnist
from medmnist import INFO
from DataClasses import PrimeGaloisField, EllipticCurve, Point
import random

import os

file_path = os.path.abspath('Aggregated model.pth')
m1 = torch.load('model1.pth')
m2 = torch.load('model2.pth')
m3 = torch.load('model3.pth')

w1 = []
w2 = []
w3 = []

# Iterate over all keys in the state dictionary
for key in m1:
    tensor = m1[key]
    # Iterate over all elements of the tensor
    for i in range(tensor.numel()):
        w1.append(int((tensor.view(-1)[i] + 1) * 100000))

# Iterate over all keys in the state dictionary
for key in m2:
    tensor = m2[key]
    # Iterate over all elements of the tensor
    for i in range(tensor.numel()):
        w2.append(int((tensor.view(-1)[i] + 1) * 100000))

# Iterate over all keys in the state dictionary
for key in m3:
    tensor = m3[key]
    # Iterate over all elements of the tensor
    for i in range(tensor.numel()):
        w3.append(int((tensor.view(-1)[i] + 1) * 100000))

inf = float("inf")

# Parameters for the Elliptic Curve being used i.e y² = x³ + 2x + 2
p = 17
field = PrimeGaloisField(p)
A = 2
B = 2
curve256 = EllipticCurve(A, B, field)
I = Point(None, None, curve256)

# G(0,1)
gx = 5
gy = 1
G = Point(gx, gy, curve256)

N = 0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551

px = 3
py = 1
P = Point(px, py, curve256)

r1 = random.randint(1, p)
r2 = random.randint(1, p)
r3 = r1 + r2

nP = Point(px, (p - py), curve256)
nG = Point(gx, (p - gy), curve256)

R1 = r1 * P
R2 = r2 * P
R3 = r3 * nP

Z1 = []
Z2 = []
Z3 = []

for i in range(100):
    Z1.append(abs(r1 + w1[i]) * G)

for i in range(100):
    Z2.append(abs(r2 + w2[i]) * G)

nr = []
for i in range(100):
    nr.append(-r3 + w3[i])

Z3 = []
for i in range(100):
    if nr[i] > 0:
        nr[i] = -nr[i]
        Z3.append(-nr[i] * nG)
    else:
        Z3.append(-nr[i] * nG)

Rsum = R1 + R2 + R3

print(Rsum)

referencePoint = []

for i in range(30):
    referencePoint.append(Z1[i] + Z2[i] + Z3[i])
    print("Reference point:", referencePoint[i])
    print()

basePoint = G

points = []
points.append(P)
check = False
temp = basePoint

while check == False:
    temp += basePoint
    if temp == I:
        check = True
    points.append(temp)

print("Length of cycle:", len(points))

experiment_name = "Project"
mlflow.set_experiment(experiment_name)

# Start MLflow run
with mlflow.start_run():
    mlflow.log_param("Base Point_x", gx)
    mlflow.log_param("Base Point_y", gy)
    mlflow.log_param("Field", p)

    iteration = 0

    for r in range(10):
        for i in points:
            if referencePoint[r] == i:
                num = points.index(i)

            iteration += 1

        print(w1[r] + w2[r] + w3[r])
        print("Reference Point matches the point", num, "in the cycle")

        mlflow.log_metric("iteration", iteration)

        if referencePoint[r] is not None:
            if referencePoint[r].x is not None:
                mlflow.log_metric("reference_point_x", referencePoint[r].x.value)
            else:
                mlflow.log_metric("reference_point_x", 0)

            if referencePoint[r].y is not None:
                mlflow.log_metric("reference_point_y", referencePoint[r].y.value)
            else:
                mlflow.log_metric("reference_point_y", 0)
        else:
            mlflow.log_metric("reference_point_x", 0)
            mlflow.log_metric("reference_point_y", 0)

# End MLflow run
mlflow.end_run()

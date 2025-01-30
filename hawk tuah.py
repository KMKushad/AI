import numpy as np
import random

def format(x):
    if x < 30:
        return "."
    
    if x > 180:
        return "@"
    
    else:
        return "o"

while True:
    start = random.randint(0, 999)
    num = input("what number: ")

    f = open(f'C:\\Users\\kmkus_4e9n0iq\\Desktop\\Coding\\AI\\mnist-dataset\\data{num}.txt', "rb")
    out = open(r'C:\Users\kmkus_4e9n0iq\Desktop\Coding\AI\output.txt', "w")

    arr = f.read()[784 * start + 0: 784 * start + 784]

    for i in arr:
        i = int(str(i), 10)

    arr = list(arr)

    for i in range(28):
        for j in range(28):
            out.write(f"{format(arr[28 * i + j])} ")
        out.write("\n")
    
    f.close()
    out.close()
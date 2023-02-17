import os
import numpy as np
import subprocess
import pandas as pd 
import csv

# def combineText(input, output):
#     open(output, 'w')
#     print("Combining: ", input)
#     command = "cat " + input + "*.txt > " + output
#     # subprocess.call(command, shell=True)
#     subprocess.run(['powershell', '-Command', command])

# myDir = 'D:\\GAN\\run_0\\stats\\'
output = 'C:\\Users\\Nate\\Desktop\\all_stats.txt'

# combineText(myDir, output + "all_stats.txt")
file = open(output,'r')
df = pd.read_csv(file, sep=", ", skiprows=0)
df.head()
print("ree")

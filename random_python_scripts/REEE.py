import matplotlib.pyplot as plt 
import numpy as np 
file = "D:\\godMode\\plot.txt"
file = open(file, 'r')
values = np.loadtxt(file)
plt.plot(values)
plt.title(f"Designs temperature\nT_init= {values[0]:.3f}, T_final = {values[-1]:.3f}, T_best = {values.min():.3f}",  fontsize=14)
plt.xlabel("Steps (#)", fontsize=16)
plt.ylabel("Temperature (K)",  fontsize=16)
plt.show()
print("REE")

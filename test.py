import numpy as np
import pandas as pd

df = pd.read_csv("test.csv")
print(df.head())
# def f(x):
#     return x**2

# x = np.linspace(0, 10, 100)
# y = f(x)

# # Calculate the gradient
# dy_dx = np.gradient(y, x)

# # Plot the function and its derivative
# import matplotlib.pyplot as plt

# plt.plot(x, y, label='f(x)')
# plt.plot(x, dy_dx, label='df/dx')
# plt.xlabel('x')
# plt.legend()
# plt.show()
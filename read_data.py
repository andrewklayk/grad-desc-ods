import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

def load_real(filename: str, x_cols: list[str], y_col: str, n: int = 1000, prop_labeled: float = 0.1):
    data = pd.read_csv(filename)
    if n > np.shape(data)[0]:
        n = np.shape(data)[0]
   # xy = data[data['Species'] != 'Iris-versicolor']
    xy = data[x_cols + [y_col]].sample(n)
    x = xy[x_cols]
    y = xy[y_col]
    n_l = int(np.ceil(n*prop_labeled))
    x_l = x[:n_l].to_numpy()
    x_u = x[n_l:].to_numpy()
    y_l = y[:n_l].to_numpy()
    y_u = y[n_l:].to_numpy()
    return x_l, x_u, y_l, y_u

x_l, x_u, y_l, y_u = load_real(filename='Iris.csv', x_cols=['SepalLengthCm', 'SepalWidthCm'], y_col='Species')
y_l[y_l != 'Iris-setosa'] = -1
y_l[y_l == 'Iris-setosa'] = 1
y_u[y_u != 'Iris-setosa'] = -1
y_u[y_u == 'Iris-setosa'] = 1
colors = ['r' if y == 1 else 'b' for y in y_l == 1]
plt.scatter(x=x_l.T[0], y=x_l.T[1], c=colors)
plt.scatter(x=x_u.T[0], y=x_u.T[1], s=1)
plt.show()
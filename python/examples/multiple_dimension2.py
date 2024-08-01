import matplotlib.pyplot as plt
import multivariate_isotonic_regression as mir
import numpy as np

from random import random

width = 20

# create some points that are roughly monotonic
# except for same added white noise
X = np.array([(i, j) for i in range(width) for j in range(width)])
y = 3 * ((X[:, 0] // 7) + (X[:, 1] // 4)) + 1 + 1.5 * np.random.rand(width ** 2)

YY, XX = np.meshgrid(range(width), range(width))

# points_to_adjacency rearranges the points roughly
# according to how many other points they dominate
adj, orig_idxs, new_idxs = mir.points_to_adjacency(X)

# we rearrange X and y to have the same ordering
X_reordered = X[new_idxs, :]
y_reordered = y[new_idxs]

group, yhat = mir.generalised_isotonic_regression(
    adj,
    y_reordered,
    loss_function = "pnorm",
    p = 1.1
)

# calculate the loss
print(mir.calculate_loss(y, yhat, loss_function = "pnorm", p = 1.1))

print(mir.is_monotonic(X_reordered, yhat))

fig = plt.figure(figsize=(25,25))
ax = fig.add_subplot(projection='3d')
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.scatter(X[:, 0], X[:, 1], y)
ax.plot_surface(XX, YY, yhat.reshape(width, width), cmap='YlOrRd_r', alpha=0.7)
plt.show()

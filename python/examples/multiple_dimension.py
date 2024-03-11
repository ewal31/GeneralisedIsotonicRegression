import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

import multivariate_isotonic_regression as mir

width = 10

# create some points that are roughly monotonic
# except for same added white noise
X = np.array([(i, j) for i in range(width) for j in range(width)])
y = X[:, 0] // 3 + np.log(X[:, 1] + 1)

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

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:, 0], X[:, 1], y)
ax.plot_surface(XX, YY, yhat.reshape(width, width), color = 'orange', alpha=0.5)
plt.show()

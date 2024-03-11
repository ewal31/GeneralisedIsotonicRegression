import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

import multivariate_isotonic_regression as mir

total_points = 10

# create some points that are roughly monotonic
# except for same added white noise
X, y = mir.generate_monotonic_points(
    total_points,
    dimensions = 1,
    sigma = 0.1
)

# points_to_adjacency rearranges the points roughly
# according to how many other points they dominate
adj, orig_idxs, new_idxs = mir.points_to_adjacency(X)

# we rearrange X and y to have the same ordering
X_reordered = X[new_idxs]
y_reordered = y[new_idxs]

group, yhat = mir.generalised_isotonic_regression(
    adj,
    y_reordered,
    loss_function = "L2"
)

# calculate the loss
print(mir.calculate_loss(y, yhat, loss_function = "L2"))

print(mir.is_monotonic(X_reordered, yhat))

plt.scatter(X, y)
plt.plot(X[new_idxs], yhat, color = 'orange')
plt.show()

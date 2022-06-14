import math

import networkx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

g = networkx.gnp_random_graph(1292, 0.02299798797581734)
degrees = [val for (node, val) in g.degree()]

bins = np.linspace(math.ceil(min(degrees)),
                   math.floor(max(degrees)),
                   20) # fixed number of bins

plt.xlim([min(degrees)-5, max(degrees)+5])

plt.hist(degrees, bins=bins, alpha=0.5)
plt.title('Random Gaussian data (fixed number of bins)')
plt.xlabel('variable X (20 evenly spaced bins)')
plt.ylabel('count')

plt.show()

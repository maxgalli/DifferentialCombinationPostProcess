from matplotlib import cm
import numpy as np
np.random.seed(0)

markers = ("v", "^", ">", "<", "s", "p", "*")

rainbow = cm.rainbow(np.linspace(0, 1, 20))
np.random.shuffle(rainbow)
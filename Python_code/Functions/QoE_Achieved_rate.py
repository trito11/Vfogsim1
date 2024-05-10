import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cumfreq
import os
import scipy.io as sio
from pathlib import Path
import sys
link=Path(os.path.abspath(__file__))
link=link.parent
link=os.path.join(link, "Functions")
sys.path.append(link)
from TD_TD import TD_TD

def QoE_Achieved_rate(rk, xk, req, Ut, UserMatrix):
    Rk = TD_TD(rk * xk, xk.shape[0], xk.shape[1])
    mean_Rk = np.mean(Rk)

    # Plotting
    a = cumfreq(mean_Rk*20, numbins=1000)
    x = a.lowerlimit + np.linspace(0, a.binsize*a.cumcount.size, a.cumcount.size)

    plt.figure(figsize=(10, 6))
    plt.plot(x, a.cumcount / x.max())
    plt.xlabel('Average achieved rate (Mbps)')
    plt.ylabel('CDF')
    plt.show()
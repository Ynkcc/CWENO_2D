


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
# make data
X1, Y1 = np.meshgrid(np.linspace(0, 1, 80), np.linspace(0, 1, 80))
X2, Y2 = np.meshgrid(np.linspace(0.00632911392405063, 1+0.00632911392405063, 80), np.linspace(0.00632911392405063, 1+0.00632911392405063, 80))
Z1=np.sin(np.pi*X1)**2*np.sin(np.pi*Y1)**2
levels = np.linspace(Z1.min(), Z1.max(), 100)
x=np.arange(10)
x=x/0.2;
np.save("save_x",x)
# plot
fig, ax = plt.subplots()

surf=ax.contourf(X1, Y1, Z1, levels=levels)

def animate(i):
    global surf
    ax.clear()
    ax.set_xlim(0.01,1)
    ax.set_ylim(0.01,1)
    if(i%2==1):
        X=X1;Y=Y1;
    else:
        X=X2;Y=Y2;
    Z=np.power(np.sin(np.pi*(X-0.02*i)),2)*np.power(np.sin(np.pi*(Y-0.02*i)),2)
    levels = np.linspace(Z1.min(), Z1.max(), 20)
    surf=ax.contourf(X, Y,Z, levels=levels)
    return surf,

anim = animation.FuncAnimation(fig, animate)
plt.show()
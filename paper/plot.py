import numpy as np
import matplotlib.pyplot as plt


def gauss(x,mu=1,sigma=.5):
    return np.exp(-(x-mu)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)

xrange = np.linspace(0,3.5,2000)

d2 = gauss(xrange, mu=1.5, sigma=.15) + gauss(xrange, mu=2, sigma=.2)
dt = xrange[1]-xrange[0]
norm = np.sum(d2)*dt

lw = 3
s=25
a = .8
plt.figure(figsize=(8,8))
ax=plt.subplot(111)
ax.plot(xrange,gauss(xrange, mu=1.75), color="black", linewidth=lw, alpha=a, label=r'$S_\mathcal{E}(\theta)$')
ax.plot(xrange,d2/norm, color="red",linewidth=lw, alpha=a, label=r'$S_\tilde{\mathcal{E}}(\theta)$')
ax.legend(prop={"size":s})
ax.set_xlabel(r'$\theta$',size=s)

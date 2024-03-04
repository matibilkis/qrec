import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
np.random.seed(1)

from matplotlib import colormaps
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize



#### Estimation of alpha
#### Estimation of alpha

a=.25
ens = range(1,2*int(1e3))
p0 = np.exp(-a**2)
est = {}
for N in tqdm(ens):
    samples = np.random.choice([0,1],N, p=[p0,1-p0],replace=True)
    est[N] = np.sqrt(-np.log(1-np.sum(samples)/N))

estimates = np.stack(list(est.values()))

ax=plt.subplot(111)
ax.plot(ens,estimates)
ax.plot(ens,1/np.sqrt(ens)+a)
ax.axhline(a,color="black")

probs={}
N = 1000
for s in range(int(1e3)):
    samples = np.random.choice([0,1],N, p=[p0,1-p0],replace=True)
    probs[s] = np.sqrt(-np.log(1-np.sum(samples)/N))
samples_estimates = np.stack(probs.values())
counts, bins = np.histogram(samples_estimates,bins=50,density=True)
ax=plt.subplot(111)
ax.bar(bins[1:], counts,width=bins[1]-bins[0])
ax.axvline(a,color="black")


a=.75
p0 = np.exp(-a**2)
probs={}
N = 1000
for s in range(int(1e3)):
    samples = np.random.choice([0,1],N, p=[p0,1-p0],replace=True)
    probs[s] = np.sqrt(-np.log(1-np.sum(samples)/N))
samples_estimates = np.stack(probs.values())
counts, bins = np.histogram(samples_estimates,bins=50,density=True)
ax=plt.subplot(111)
ax.bar(bins[1:], counts,width=bins[1]-bins[0])
ax.axvline(a,color="black")



##### NOISE-LESS

def p(alpha,n):
    """
    p(n|alpha), born rule
    """
    pr = np.exp(-(alpha)**2)
    return [pr, 1-pr][n]

def Perr(beta,alpha=0.4):
    ps=0
    for n in range(2):
        ps+=np.max([p(sgn*alpha + beta,n) for sgn in [-1,1]])
    return 1-ps/2

def model_aware_optimal(betas_grid, alpha=0.4):
    #### Landscape inspection
    mmin = minimize(Perr, x0=-alpha, args=(alpha),bounds = [(np.min(betas_grid), np.max(betas_grid))])
    p_star = mmin.fun
    beta_star = mmin.x[0]
    return mmin, p_star, beta_star

betas_grid = np.linspace(-1.5,1.5,200)
_,_, b025 = model_aware_optimal(betas_grid, alpha=0.25)
_,_, b075 = model_aware_optimal(betas_grid, alpha=0.75)

ax=plt.subplot(211)
ax.plot(betas_grid,1.-np.array([Perr(b,alpha=0.7) for b in betas_grid]),color="red")
ax.plot(betas_grid,1.-np.array([Perr(b,alpha=0.75) for b in betas_grid]),color="blue")
ax.plot(betas_grid,1.-np.array([Perr(b,alpha=0.8) for b in betas_grid]),color="black")
ax.axvline(b075,color="green")
ax=plt.subplot(212)
ax.plot(betas_grid,1.-np.array([Perr(b,alpha=0.2) for b in betas_grid]),color="red")
ax.plot(betas_grid,1.-np.array([Perr(b,alpha=0.25) for b in betas_grid]),color="blue")
ax.plot(betas_grid,1.-np.array([Perr(b,alpha=0.3) for b in betas_grid]),color="black")
ax.axvline(b025,color="green")

alphas = np.linspace(1e-4,1.5,15)
opts = []
for a in alphas:
    f = model_aware_optimal(betas_grid, alpha=a)
    opts.append(f[1:])
opts = np.squeeze(opts)

plt.figure(figsize=(7,4))
ax=plt.subplot(211)
ax.plot(alphas,1-opts[:,0])
ax=plt.subplot(212)
ax.plot(alphas,opts[:,1])


##############

np.abs(1 + 1j)

def p(alpha,n):
    """
    p(n|alpha), born rule
    """
    pr = np.exp(-np.abs(alpha)**2)
    return [pr, 1-pr][n]

def Perr(beta,noi=0.5,alpha=0.4):
    ps=0
    pr=[noi,1-noi]
    for n in range(2):
        ps+=np.max([p(sgn*alpha + beta,n)*pr[i] for i,sgn in enumerate([-1,1])])
    return 1-ps

def model_aware_optimal(betas_grid, noi=.5, alpha=0.4):
    #### Landscape inspection
    mmin = minimize(Perr, x0=-alpha, args=(noi, alpha),bounds = [(np.min(betas_grid), np.max(betas_grid))])
    p_star = mmin.fun
    beta_star = mmin.x[0]
    return mmin, p_star, beta_star

def setleg(leg,lw=2):
    for legobj in leg.legendHandles:
        legobj.set_linewidth(lw)


Nnoi, Na = 5, 7
betas_grid = np.linspace(-1.5,0,200)
alphas = np.linspace(.25,.75,Na)
noises = np.linspace(0,.5,Nnoi)
opts = []
for noi in noises:
    for a in alphas:
        f = model_aware_optimal(betas_grid, noi=noi, alpha=a)
        opts.append(f[1:])
opts = np.squeeze(opts).reshape((Nnoi, Na,2))



colormap = plt.cm.ocean #, Set1,Paired
colors = [colormap(i) for i in np.linspace(.2, .8,len(noises))]
plt.figure(figsize=(7,7))
plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(noises)))))
ax=plt.subplot(211)
ax.set_xticks([])
for n,noi in enumerate(noises):
    ax.plot(alphas,1-opts[n,:,0],label=str(np.round(noi,2)), color=colors[n],linewidth=2)
ax.set_ylabel(r'$P_s$',size=30)
leg = ax.legend(prop={"size":15},loc="lower right")
setleg(leg,lw=4)
ax=plt.subplot(212)
for n,noi in enumerate(noises):
    ax.plot(alphas,opts[n,:,1], label=str(np.round(noi,2)),color=colors[n],linewidth=2)
ax.set_xlabel(r'$\alpha$',size=30)
ax.set_ylabel(r'$\beta^*$',size=30)
leg=ax.legend(prop={"size":15},loc="lower right")
setleg(leg,lw=4)



_,_, b025_ideal = model_aware_optimal(betas_grid, noi=.5, alpha=0.25)
_,_, b025_noise = model_aware_optimal(betas_grid, noi=.4, alpha=0.25)

betas_gridd = np.linspace(-1,0,10)
plt.figure(figsize=(7,7))
ax=plt.subplot(211)
ax.plot(betas_grid,1.-np.array([Perr(b,noi=.5,alpha=0.25) for b in betas_grid]),color="green", label=r'$p_0 = 0.5$',linewidth=3)
ax.plot(betas_grid,1.-np.array([Perr(b,noi=.4,alpha=0.25) for b in betas_grid]),color="red", label=r'$p_0 = 0.4$',linewidth=3)
ax.scatter(betas_gridd,1.-np.array([Perr(b,noi=.5,alpha=0.25) for b in betas_gridd]),color="green")
ax.scatter(betas_gridd,1.-np.array([Perr(b,noi=.4,alpha=0.25) for b in betas_gridd]),color="red")
ax.axvline(b025_ideal,color="green")
ax.axvline(b025_noise,color="red")
leg=ax.legend(prop={"size":15},loc="lower right")
setleg(leg,lw=4)

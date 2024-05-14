# %%
import torch
import gvar as gv
import numpy as np
import lsqfit as lsf
from plot_settings import *
from general_plot_funcs import errorbar_plot, fill_between_plot


######## ADJUST HERE ##########
if_correlation = False
data_quality = "high" #*: "high" and "low"
each_sample = False



def extrapolate_and_ft(z_dep_gv, fit_z_min, fit_z_max, tail_end, zmax_keep):
    # * z_dep_ls is real part only

    zmax = len(z_dep_gv)

    z_array = np.arange(zmax)

    priors = gv.BufferDict()
    priors["b"] = gv.gvar(1, 10)
    priors["c"] = gv.gvar(0, 10)
    priors["d"] = gv.gvar(0, 10)
    priors["log(n)"] = gv.gvar(0, 10)
    priors["log(m)"] = gv.gvar(0, 10)
    # priors["n"] = gv.gvar(0, 10)
    # priors["m"] = gv.gvar(0, 10)
    priors["s"] = gv.gvar(0, 10)

    def fcn(x, p):
        #todo poly * exp * power
        # return p["b"] * np.sin( p["c"] * x + p["d"] ) * np.exp(-x * p["m"]) / (x ** p["n"])
        return ( p["b"] + p["c"] * x + p["d"] * x**2 ) * np.exp(-x * p["m"]) / (x ** p["n"])
        # return p['b'] + p['c'] * np.exp( p['d'] * np.tanh( p['n'] * x + p['m'] ) )
        # return p['b'] * np.sin( p['c'] * np.tanh( p['d'] * x + p['n'] ) + p['m'] ) + p['s']
    
    # to fit in extrapolation
    fit_mask = (z_array >= fit_z_min) & (z_array <= fit_z_max)
    fit_z_array = z_array[fit_mask]

    # to keep the original data
    #* keep the data points smaller than zmax_keep * a
    keep_mask = z_array < zmax_keep

    z_dep_ext = []
    fit_gv = z_dep_gv[fit_mask]

    fit_res = lsf.nonlinear_fit(
        data=(fit_z_array, fit_gv), fcn=fcn, prior=priors, maxit=10000
    )

    print(fit_res.format(100))

    # complete the tail
    z_gap = abs(z_array[1] - z_array[0])
    # * start to apply extrapolation from the first point larger than zmax_keep
    tail_array = np.arange(z_array[keep_mask][-1] + z_gap, tail_end, z_gap) 
    z_dep_tail = fcn(tail_array, fit_res.p)

    # concatenate the original part and the tail part
    z_array_ext = np.concatenate((z_array[keep_mask], tail_array))
    z_dep_ext = np.concatenate((z_dep_gv[keep_mask], z_dep_tail))

    return z_array_ext, z_dep_ext

if data_quality == "low":
    z_dep_samples = gv.load("z_dep_collection_cut_222.dat")['re_p5_b8']
elif data_quality == "high":
    z_dep_samples = gv.load("z_dep_collection_cut_222.dat")['re_p3_b2']

print(np.shape(z_dep_samples))

# np.savetxt("zdep_sample.csv", z_dep_samples)

z_dep_gv = gv.dataset.avg_data(z_dep_samples, bstrap=True)

errorbar_plot(np.arange(len(z_dep_gv)), gv.mean(z_dep_gv), yerr=gv.sdev(z_dep_gv), title="data check", save=False)


# %%
if if_correlation == False:
    z_dep_gv = gv.gvar( gv.mean(z_dep_gv), gv.sdev(z_dep_gv) )

z_array_ext, z_dep_ext = extrapolate_and_ft(z_dep_gv, 8, 20, 20, 10)

errorbar_plot(z_array_ext, gv.mean(z_dep_ext), yerr=gv.sdev(z_dep_ext), title="extrapolation check", save=False)


if each_sample == True:
    data_x = []
    data_y = []
    N_samp = len(z_dep_samples)
    for idx in range(N_samp):
        x_ls = np.arange(5, 20)
        y_ls = z_dep_samples[idx][5:20]
        data_x += list(x_ls)
        data_y += list(y_ls)

elif each_sample == False:
    data_x = np.arange(5, 20)
    data_y = gv.mean(z_dep_gv[5:20])


data_x = np.array(data_x)
data_y = np.array(data_y)

data_x = torch.from_numpy( data_x.reshape(-1,1) )
data_y = torch.from_numpy( data_y.reshape(-1,1) )

# print(data_x, data_y)


# %%
#! Symbolic regression
from kan import *
# create a KAN: 1D inputs, 1D output, and 4 hidden neurons. cubic spline (k=3), 4 grid intervals (grid=3).
model = KAN(width=[1,3,1], grid=3, k=3, seed=0)

# %%
# create dataset f(x) = exp(sin(pi*x))
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]))
dataset = create_dataset(f, n_var=1)
dataset['train_input'].shape, dataset['train_label'].shape

# %%
dataset = {}  # Define the dataset variable
dataset['train_input'], dataset['train_label'] = data_x, data_y
dataset['test_input'], dataset['test_label'] = data_x, data_y


# %%
# plot KAN at initialization
model(dataset['train_input']);
model.plot(beta=100)

# %%
# train the model
model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10.);
model.plot()


# %%
# model = model.prune()
# model(dataset['train_input'])
# model.plot()


# %%
model.train(dataset, opt="LBFGS", steps=50);
model.plot()

# %%
mode = "auto" # "manual"

if mode == "manual":
    # manual mode
    model.fix_symbolic(0,0,0,'sin');
    model.fix_symbolic(0,1,0,'x^2');
    model.fix_symbolic(1,0,0,'exp');
elif mode == "auto":
    # automatic mode
    lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin']#,'abs']
    model.auto_symbolic(lib=lib)

# %%
# model.train(dataset, opt="LBFGS", steps=20);
model.symbolic_formula()[0][0]

# %%

from ase.db import connect
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import numpy as np
import os
import json
from tqdm import tqdm
from time import sleep
from ase.visualize import view
from ase.io import read, write, Trajectory, lammpsdata
from ase.calculators.eam import EAM
import time
import sys
import shutil
from ase.calculators.emt import EMT

# mishin = EAM(potential='AuCu_Zhou04.eam.alloy')
# mishin.write_potential('new.eam.alloy')
# mishin.plot()
#
# print("hha")

# with connect('fff.db') as con:
#     print("what the fuck is this?!?!?!")

# a = np.array([1,2,3,4])
# b = a*2
# c = a*3
#
# data = [a,b,c]
# print(data)

# a = [1,2,3]
# b = []
# b.append(a)
# b.append(a)
# print(b)
# flat_lims = np.logspace(np.log10(0.4), np.log10(0.01), 5)
# flat_lims = 1 - flat_lims
# aa  = np.logspace(np.log10(0.1), np.log10(0.02), 5)
#
# for idx, (fl, a) in enumerate(zip(flat_lims, aa)):
#     print(idx, fl, a)


# a = 1-a
# print(a)
#
# pbar = tqdm(total=0.9)
#
# for i in range(9):
#     # pbar.update(0.1)
#     pbar.n = i/10
#     pbar.refresh()
#     sleep(.5)
# pbar.close()
#
# atoms = read("/Users/Michael_wang/Documents/venkat/clease/testGetE/Au2Cu14_sdfas_9/bulk/results_h/Au2Cu14-0.16.traj",  index=":")
# view(atoms)

############# chuan ################

# x = np.linspace(-0.5, 1.5, 1000
# from sympy import *
# var('x y')
# plot_implicit((Eq(pow(x,2)+x)*pow(y,2)+(10*pow(x,2)-20*x)*y+25*pow(x^2)-125*x), xlim=(-0.5, 0.5), ylim=(-6, 40))

##################### plot single Metadynamics #######################
# prefix = 'ASE-EAM-aucu_metadyn-300-0'
#
# def animate(i):
#     with open(prefix + '.json', 'r') as f:
#         meta_dyn_data = json.load(f)
#
#     x = meta_dyn_data['betaG']['x']
#     y = meta_dyn_data['betaG']['y']
#     plt.cla()
#
#     plt.plot(x, -np.array(y))
#
# ani = FuncAnimation(plt.gcf(), animate, interval=1000)
# plt.show()


##################### plot multiple Metadynamics #######################

# plt.figure()
# for t in [10, 30, 50, 70, 90, 120]:
#     with open('aucu_metadyn-' + str(t) + '-2.json', 'r') as f:
#         meta_dyn_data = json.load(f)
# for t in range(3):
#     with open('1x-aucu_metadyn-30-' + str(t) + '.json', 'r') as f:
#         meta_dyn_data = json.load(f)
#
#     x = meta_dyn_data['betaG']['x']
#     y = meta_dyn_data['betaG']['y']
#     plt.plot(x, (np.array(y)-y[0])/(y[-1]-y[0])-0.05*t, label=str(t))
#
# plt.legend()
# plt.show()

#### visualize ###

# dbName = "0.75.db"
# db = connect(dbName)
# i = 0
# for row in db.select(""):
#     view(row.toatoms())
#     i += 1
#     if i == 30:
#         exit(0)


# fp = np.memmap(fname, dtype='float64', mode='w+', shape=(150000, 6))
# fp[:] = np.random.rand(150000, 6)
# fp.flush()


# with open("/Users/Michael_wang/haha.csv", 'w') as f:
#     # create the csv writer
#     writer = csv.writer(f)
#     data = np.random.rand(150000 * 6)
#     writer.writerow(data)

##########
# a = np.array([1.5,2,1,5,8,7,10])
# b = np.array([15, 20,10,50,80,70,100])
# aBin = np.linspace(0,11,6)
# finals = np.zeros(6)
# idx = np.digitize(a,aBin)
# for j in range(1, len(aBin)):
#     finals[j-1] = np.average(b[idx == j])
#
#
# exit()

# i, j = 0, 0
# while j < len(bb):
#     while i < len(aa):
#         while j <len(bb):
#             a = aa[i]
#             b = bb[j]
#             print(a+b)
#             j += 1
#             # break
#         j = 0
#         i += 1
#         print()
#     # i = 0
#     j += 1

# order = np.array([1, 0], dtype=bool)
# # order = ~order
# # print(np.sum([len(aa), len(bb)]*order))
# for i_tmp in range(np.sum([len(aa), len(bb)]*order)):
#     for j_tmp in range(np.sum([len(aa), len(bb)]*~order)):
#         a = aa[np.sum([i_tmp, j_tmp] * order)]
#         b = bb[np.sum([i_tmp, j_tmp] * ~order)]
#         print(a+b, "a =", a, " b =", b)
#     print()

#####################     ########################

# # from ase.build import bulk
# # from ase.calculators.lammpslib import LAMMPSlib
# # from ase.db import connect
# # from ase.visualize import view
# # from ase.calculators.emt import EMT
# # from ase.calculators.eam import EAM
# # from ase.optimize import BFGS
# # from ase.constraints import UnitCellFilter

# # db = connect("toy.db")
# # atoms = None
# # for row in db.select(converged=True):
# #     atoms = row.toatoms()

# from ase import Atom, Atoms
# from ase.neighborlist import NeighborList

# # Create an example Atoms object
# atoms = Atoms([Atom('H', [0, 0, 0]),
#                Atom('O', [1, 0, 0]),
#                Atom('H', [2, 0, 0]),
#                Atom('O', [3, 0, 0])])

# # Define the cutoff radius for neighbors
# cutoff_radius = 1

# # Create the NeighborList object
# nl = NeighborList(cutoffs=[cutoff_radius / 2] * len(atoms),
#                   self_interaction=False,
#                   bothways=True)

# # Update the neighbor list
# nl.update(atoms)

# # Choose an atom index to get its neighbor list
# atom_index = 1

# # Get the indices of neighbors for the chosen atom
# neighbor_indices, _ = nl.get_neighbors(atom_index)

# # Print the neighbor atoms
# print("Neighbors of atom", atom_index)
# for neighbor_index in neighbor_indices:
#     print("Atom index:", neighbor_index)
#     print("Atom symbol:", atoms[neighbor_index].symbol)
#     print("Atom position:", atoms[neighbor_index].position)
#     print("----------")

# exit()

#
# view(atoms)
# exit()
#
# # calc = EMT()
# calc = EAM(potential='AuCu_Zhou04.eam.alloy')
# atoms.calc = calc
# print(atoms.get_potential_energy())
# # ucf = UnitCellFilter(atoms)
# opt = BFGS(atoms, logfile=None)
# opt.run(fmax=0.001)
# print(atoms.get_potential_energy())

################ plot MF free energy ################

# N = 100
# q = 6
# J = -0.005
# JJ = 0.01
# h = 8
# kB = 8.617333262e-5
# T = 10
# Tc = 2.269*J/kB
#
# m = np.linspace(-1, 1, 100)
# hEff = 0*h + J * q * m + JJ * q * m**2
#
# hEff2 = 0*h + J * q * m - JJ * q * m**2
#
# F = - N * q * J * m**2 / 2  -0* N * q * J * m**3 / 2 - N * kB * T * np.log(2) \
#     + N * kB * T * np.log(np.cosh(hEff/kB/T)) \
#     +  N * kB * T * np.log(np.cosh(hEff2/kB/T)) - h*m
#
# plt.plot(m, F)
# plt.show()
#
# exit()


import subprocess
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
subprocess.run(["nequip-train", "/Users/Michael_wang/Documents/venkat/cleaseASEcalc/minimal_eng.yaml"])


############################ plot free energy ######################
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator
# import numpy as np
#
#
# # TInit, TFinal = 3e3, 1.5e5
# # muInit, muFinal = 30, -35
# # dT, dMu = 3000, -1
# # Ts = np.arange(TInit, TFinal+1e-5, dT)
# # mus = np.arange(muInit, muFinal+1e-5, dMu)
#
# # data = np.load("XTablelele.npy", allow_pickle=True)
#
# # print(len(data[0]))
#
# # plt.plot(mus, data[0])
# # plt.ylabel("Concentration")
# # plt.xlabel("Chem_pot")
# # plt.show()
#
# # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#
# # Make data.
#
#
# a = np.load("toy-phiTable-0.0.npy")
# b = np.load("toy-phiTable-1.0.npy")
# # b = b[:, 0:65]
# x = np.load("toy-XTable-0.0.npy")
#
# TInit, TFinal = 3e3, 3e5
# muInit, muFinal = -35, 30
# dT, dMu = 3000, 1
# TTs = np.arange(TInit, TFinal+1e-5, dT)
# mus = np.arange(muInit, muFinal+dMu*1e-5, dMu)
#
# # plt.plot(a[20])
# # plt.show()
#
#
# mus, Ts = np.meshgrid(mus, TTs)
# # Z = Ts + mus
#
# # a = a + mus*x
#
# print(a.shape)
# # print(b.shape)
# print(Ts.shape)
# print(mus.shape)
#
# upTo = 80
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(Ts[:upTo], mus[:upTo], a[:upTo], cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# surf = ax.plot_surface(Ts[:upTo], mus[:upTo], np.flip(b, axis=1)[:upTo], cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# plt.show()
#
# exit()
#
# #
# #
# # print(np.max(a, axis=0))
# #
# # exit()
# #
# # xs = np.load("X4C.npy")
# # cs = np.load("C4C.npy")
# #
# # plt.figure()
# # plt.plot(xs[40])
# # plt.figure()
# # plt.plot(cs[40])
# # plt.show()
# #
# # finalCs = np.zeros((xs.shape[0], 50))
# # xBins = np.linspace(0.5, 0.75, 51)
# # for i, x in enumerate(xs):
# #     xIdx = np.digitize(x, xBins)
# #     for j in range(1, len(xBins)):
# #         finalCs[-(i+1)][j-1] = np.average(cs[i][xIdx == j])
# #
# # plt.imshow(finalCs)
# # plt.show()
# #
# #
# #
# #
# # # plt.figure()
# # # plt.plot(x[20])
# # # plt.figure()
# # # plt.plot(c[20])
# # # plt.show()
# # # pass
# # #
# # # x = np.array([0.2, 6.4, 3.0, 1.6])
# # # bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
# # # inds = np.digitize(x, bins)

############# test CV ############
# def find_CV(ys, p):
#     yLength = len(ys)
#     x = np.arange(len(ys))
#     CV = 0
#     for i in range(yLength):
#         fit = np.polyfit(np.delete(x, i), np.delete(ys, i), p)
#         fn = np.poly1d(fit)
#         CV += (fn(i) - ys[i]) ** 2
#     CV /= yLength
#     return CV
#
#
# def dis_detector(ys, newY, small_value=0.001):
#     yLength = len(ys)
#     if yLength < 3: return 0
#     if yLength > 20: return dis_detector(ys[-20:], newY, small_value)
#     bestP = 0
#     bestCV = np.inf
#     for p in range(yLength - 1):
#         CV = find_CV(ys, p)
#         if CV + 1e-15 < bestCV:
#             bestCV = CV
#             bestP = p
#
#     print(bestP)
#
# # y = [1,2,3,4,5,6,7,8,9]
# x = np.linspace(0, 10, 10)
# y = 0*x + 3 + np.random.rand(10)
# # dis_detector(y, 3235243)
# print(y)
#
# # print(np.delete(a, 1))
# # print(np.delete(a, 3))
# exit(0)

##################### plot formation energy #######################

# temps = np.linspace(50, 600, 12)
# for temp in temps:
#     shutil.copyfile("aucu-Mar14.db", "aucu-Mar17-" + str(temp) + ".db")
#
# exit(0)

# a = np.linspace(5000, 200, 100)
# print(np.sum(a))
# exit(0)
#
# db_name = "aucu-Mar17.db"
# db = connect(db_name)
# i = 0
# cons = []
# Es = []
# # db.delete(range(4,23))
# # for row in db.select(""):
# #     print(row)
# #
# # exit(0)
#
# for row in db.select(struct_type="final"):
#     atoms = row.toatoms()
#     energy = atoms.calc.results['energy']/len(atoms.numbers)
#     con = np.count_nonzero(atoms.numbers != 29)/len(atoms.numbers)
#     Es.append(energy)
#     cons.append(con)
#     i += 1
#
# print(i)
# assert cons[0] == min(cons)
# assert cons[-1] == max(cons)
# low = Es[0]
# high = Es[-1]
# Es = np.array(Es)
# cons = np.array(cons)
# Es = Es - cons*high - (1-cons)*low
# plt.plot(cons, Es)
#
#
# plt.show()


##################### plot Metadynamics from Arjuna #######################

# temps = np.linspace(10, 800, 40)
#
# data = np.load("metaData.npy")
# for idx, d in enumerate(data):
#     plt.plot(d)
#     plt.title(str(temps[idx]))
#     plt.show()

######################### heat map ############################
# Y = np.logspace(np.log10(1000), np.log10(20), 100)
# X = np.linspace(0.01, 0.5, 40)
# data = np.load('Cdata.npy', allow_pickle=True)
#
# critical_temps = []
# for d in data:
#     critical_temps.append(Y[np.argmax(d)])
#
# data = data.T
#
#
# plt.plot(X, critical_temps)
# plt.ylim([100, 800])
# plt.show()
#
#
# plt.pcolor(X,Y,data)
# plt.ylim([100, 800])
# plt.show()

#################################################################

#
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#
# surf = ax.plot_surface(X, Y, data, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
#
# # plt.imshow(np.array(data.T, dtype=float))
# # plt.yscale('log')
# plt.show()



# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator
# import numpy as np
#
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#
# # Make data.
# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)
#
# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
#
# # Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# # A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')
#
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
#
# plt.show()


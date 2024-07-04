import sys
sys.path.append("/nfs/turbo/coe-venkvis/ziqiw-turbo/PD/solid/clease")

import json
import logging
import sys, os, time
import yaml

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# from lammps import lammps

from ase.db import connect
from ase.calculators.emt import EMT
from ase.calculators.eam import EAM
from ase.calculators.lammpslib import LAMMPSlib
from ase.visualize import view
from ase import Atom, Atoms
from ase.build import bulk
from ase.data import atomic_numbers
from ase.io.trajectory import TrajectoryWriter
from ase.io import write, read

from clease.settings import Concentration
from clease.structgen import NewStructures
from clease.settings import CEBulk
from clease.calculator import attach_calculator
from clease.montecarlo import Montecarlo, constraints, SGCMonteCarlo, BinnedBiasPotential, MetaDynamicsSampler
from clease.montecarlo.observers import ConcentrationObserver, MoveObserver, CorrelationFunctionObserver
from clease.tools import update_db
from clease import Evaluate
import clease.plot_post_process as pp

from chgnet.model.model import CHGNet
from chgnet.model.dynamics import MolecularDynamics, StructOptimizer, CHGNetCalculator
from pymatgen.core import Structure

from drawpd import LTE, MF

kB = 8.617333262e-5
np.set_printoptions(precision=3, suppress=True)

"""
⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽
############# DEBUG == 0 #############
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
Z test acc
phi_LTE
normal scan:
    T, mu
    convergence progress: i = ?000, coverged after ?

⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽
############# DEBUG == 1 #############
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
normal scan:
    print convergerd atoms to termial
    array of qautities used for disconti test [oldYs, newY]

disconti_test:
    fitted function
    best order of fit
    new predictions
    little v
    sigma
    @ what mu, z score = ?

⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽
############# DEBUG == 2 #############
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
normal scan:
    PLOT converging history

disconti_test:
    PLOT data, fit, and predictions

"""
with open('LiMg-example.yaml', 'r') as file:
    options = yaml.safe_load(file)


DEBUG = options["EMC"]["DEBUG"]


ZTestAcc = 0.01
Z = stats.norm.ppf(1-ZTestAcc/2)
print("Running with z test acc of ", 1-ZTestAcc, " with Z star = ", Z)


# consIdx = sys.argv[1]
# conss =  np.linspace(0.3, 0.5, 10)
# cons = conss[int(consIdx)]

# rootDir = 'ConsIdx'+consIdx
rootDir = "pyEmc_test"



# logging.basicConfig(filename=os.path.join(rootDir, 'CLEASE-heatCap.log'), level=logging.INFO)
# ranSeed = 926 #int(np.random.random()*1000)
# print("rs",ranSeed)
# np.random.seed(ranSeed)  # Set a seed for consistent tests

if "LAMMPS" in options:
        from ase.calculators.lammpslib import LAMMPSlib
        print("LAMMPS calc!")
        ASElammps = LAMMPSlib(**options["LAMMPS"])

# cmds = ["pair_style eam/alloy",
#         "pair_coeff * * AuCu_Zhou04.eam.alloy Cu Au"]

# ASElammps = LAMMPSlib(lmpcmds=cmds, keep_alive=True, log_file=os.path.join(rootDir, "LAMMPSlog.log"))
# cmds = ["pair_style eim",
#         "pair_coeff * * Na Li /Users/Michael_wang/Documents/venkat/cleaseASEcalc/ffield.eim Na Li"]
# rootDir = "/Users/Michael_wang/Documents/venkat/cleaseASEcalc/LiNa"
# ASElammps = LAMMPSlib(lmpcmds=cmds, keep_alive=True, log_file=os.path.join(rootDir, "LAMMPSlog.log"))

# cmds = ["pair_style meam",
#         "pair_coeff * * /Users/Michael_wang/Documents/venkat/MEAM/library.meam Li Mg /Users/Michael_wang/Documents/venkat/MEAM/LiMg.meam Li Mg"]
# rootDir = "/Users/Michael_wang/Documents/venkat/cleaseASEcalc/LiMg"
# ASElammps = LAMMPSlib(lmpcmds=cmds, atom_types={'Li':1, 'Mg':2}, keep_alive=True, log_file=os.path.join(rootDir, "LAMMPSlogMg.log"))
# lmp = lammps()

# print(ASElammps, lmp)


# conc = Concentration(basis_elements=[['Li', 'Mg']])
# db_name = "LiMg/LiMg-Sep8-first-Batch-hcp.db"
# MCsettings = CEBulk(crystalstructure='hcp',
#                   a=3.17,
#                   c=5.14,
#                   supercell_factor=64,
#                   concentration=conc,
#                   db_name=db_name,
#                   max_cluster_dia=[5.5,5.5,5.5],
#                   basis_func_type="polynomial")
# with open(db_name + "-eci.json") as f:
#     eci = json.load(f)

# conc = Concentration(basis_elements=[['Li', 'Mg']])
# db_name = "LiMg/LiMg-Sep8-first-Batch-bcc.db"
# MCsettings = CEBulk(crystalstructure='bcc',
#                   a=4.33,
#                   supercell_factor=64,
#                   concentration=conc,
#                   db_name=db_name,
#                   max_cluster_dia=[7,7,7],
#                   basis_func_type="polynomial")
# with open(db_name + "-eci.json") as f:
#     eci = json.load(f)

# db_name = "LiNa-demo.db"
# conc = Concentration(basis_elements=[['Li', 'Na']])

# MCsettings = CEBulk(crystalstructure='bcc',
#                   a=3.9,
#                   supercell_factor=64,
#                   concentration=conc,
#                   db_name=db_name,
#                   max_cluster_dia=[7,7,7],
#                   basis_func_type="polynomial")
# with open("LiNa-eci.json") as f:
#     eci = json.load(f)

# conc = Concentration(basis_elements=[['Au', 'Cu']])

# curr_db_name = "aucuEAMUnrelaxed.db"
# MCsettings = CEBulk(crystalstructure='fcc',
#                   a=3.8,
#                   supercell_factor=27,
#                   concentration=conc,
#                   db_name=curr_db_name,
#                   max_cluster_dia=[6.5, 4.0, 4.0])
# with open('aucu-Mar17.db-eci.json') as f:
#     eci = json.load(f)
# with open('aucu-May31.db-eci.json') as f:
#     eci = json.load(f)

# curr_db_name = "toy-fero.db"
# MCsettings = CEBulk(crystalstructure='sc',
#                   a=3,
#                   supercell_factor=64,
#                   concentration=conc,
#                   db_name=curr_db_name,
#                   max_cluster_dia=[4,4,4],
#                   basis_func_type="polynomial")
# bf_ls = MCsettings.basis_functions
# print(bf_ls)
# with open('toy-fero-eci.json') as f:
#     eci = json.load(f)

# from clease.tools import species_chempot2eci
# eci = species_chempot2eci(bf_list=bf_ls, species_chempot={"Cu":-1, "H":2, "O":3.33})


def get_c1s(eci):
    singletECI = {}
    for k, v in eci.items():
        if k[:2] == "c1":
            singletECI[k] = v

    c1s = np.zeros(len(singletECI))
    for k, v in singletECI.items():
        c1s[int(k[3:])] = v

    return c1s

def eci2chem_pot(bf_ls, eci):
    c1s = get_c1s(eci)

    mat = np.zeros((len(bf_ls), len(bf_ls)))

    for i in range(len(bf_ls)):
        for j, sp in enumerate(bf_ls[i]):
            if j == 0:
                continue
            mat[j-1, i] = bf_ls[i][sp]

    chem_pots = mat.dot(c1s)
    return chem_pots

# print(get_c1s(eci))
# print(eci2chem_pot(bf_ls, eci))

# exit()

def generate_gs_structures(MCsettings, eci, starting=None, startConc=np.array([0.5, 0.5])):
    if starting == None:
        atoms = MCsettings.prim_cell.copy()
        atoms *= (6,6,6)
        MCsettings.set_active_template(atoms)

        new_struct = NewStructures(MCsettings)

        num_atoms_in_basis = [len(indices) for indices in MCsettings.index_by_basis]
        print(num_atoms_in_basis)
        num_to_insert = MCsettings.concentration.conc_in_int(num_atoms_in_basis, startConc)
        print(num_to_insert)
        MCns = new_struct._random_struct_at_conc(num_to_insert)
        print(MCns)

        MCns = attach_calculator(MCsettings, atoms=MCns, eci=eci)
    else:
        MCns = attach_calculator(MCsettings, atoms=starting, eci=eci)

    view(MCns)

    mcSGC = SGCMonteCarlo(MCns, 300, symbols=['Li', 'Mg'], observe_singlets=True)

    def get_conc(atoms):
        return np.sum(atoms.numbers == 79) / len(atoms.numbers)

    E = []
    numOfAu = []
    numOfAu.append(get_conc(MCns))
    for i in range(1000):
        if not i % 1000:
            print(i)
        mcSGC.run(steps=100, chem_pot={'c1_0': -0.4027})
        E.append(mcSGC.get_thermodynamic_quantities()['singlet_energy'])
        numOfAu.append(get_conc(MCns))

    print(MCns)
    print(E[-1])
    view(MCns)
    plt.figure()
    plt.plot(numOfAu)
    plt.figure()
    plt.plot(E)
    plt.show()

    MCns.write("0.00.4.db")

# atoms = MCsettings.prim_cell.copy()
# atoms.numbers[0] = 11
# atoms *= (4,4,4)
# atoms.numbers[0] = 3
# changeToCu = [0,1,2,3,4,5,6,7]
# for i in changeToCu:
#     atoms.numbers[i] = 29
# atoms *= (3,3,3)
# atoms.write("LiNa/LiNa-64-1.0-EIM.db")
# view(atoms)

# db = connect("LiNa/LiNa-64-0.0-EIM.db")
# atoms = None
# for row in db.select(""):
#     atoms = row.toatoms()
# view(atoms)

# # atoms = attach_calculator(MCsettings, atoms=atoms, eci=eci)
# # print(atoms.get_potential_energy())

# generate_gs_structures(MCsettings, eci, atoms, startConc=np.array([0.0, 1.0]))

# exit(0)

# db = connect("Li-0.0.db")
# atoms = None
# for row in db.select(""):
#     atoms = row.toatoms()

# atoms = attach_calculator(MCsettings, atoms=atoms, eci=eci)
# print(atoms.get_potential_energy())
# exit(0)

def get_conc(atoms):
    return np.sum(atoms.numbers == 3) / len(atoms.numbers) * 2 - 1

def plot_stuff(x, stuff, myTitle = None, breakingpt = None):
    for i, s in enumerate(stuff):
        plt.figure()
        try:
            plt.plot(x, s, "-o")
        except:
            plt.plot(s, "-o")
        if len(myTitle) == 1:
            plt.title(myTitle[0])
        elif myTitle:
            plt.title(myTitle[i])
        if breakingpt != None:
            for b in breakingpt:
                plt.vlines(b, ymin=min(s), ymax=max(s))
    # plt.show()

def find_CV(x, ys, p):
    yLength = len(ys)

    CV = 0
    for i in range(yLength):
        fit = np.polyfit(np.delete(x, i), np.delete(ys, i), p)
        fn = np.poly1d(fit)
        CV += (fn(i) - ys[i]) ** 2
    CV /= yLength
    return CV

class equilibrium_detector:
    def __init__(self, granularity, nb_bins, prec, patience):
        # WARNING! Do not set patience to be larger than floor(nb_bins/2)
        self.patience = patience
        assert not nb_bins%2, "nb_bins must be a even number"
        self.nb_bins = nb_bins
        self.prec = prec
        # POI: Property of Interest
        self.POISum = 0
        self.POI2Sum = 0
        self.POITmpRecord = np.zeros(granularity)
        # +1 because curr_bin initialize to be 1, be careful when using np.average, due to the +1
        self.POIBins = np.zeros(nb_bins+1) # store POISum
        self.POI2Bins = np.zeros(nb_bins+1)  # store POI^2Sum
        self.corrBins = np.zeros(nb_bins+1)
        self.otherSum = 0
        self.otherBins = np.zeros(nb_bins+1)

        self.finalPOI = np.inf
        self.finalOther = np.inf

        self.granularity = granularity
        self.curr_corr_len = int(granularity / 2)
        self.cnt = 0
        # cur_bin start from 1 since we always leave 0th bin to be 0, since that's with bin with nothing sumed
        self.curr_bin = 1

        self.converged = False
        self.equil = 0
        self.steps = 0


    def reset(self):
        self.__init__(granularity=self.granularity, nb_bins=self.nb_bins, prec=self.prec)

    def get_POI(self):
        return self.finalPOI

    def get_other(self):
        return self.finalOther

    def new_data(self, POI, other):
        POI = POI + (np.random.rand() - .5)*0.001
        self.POITmpRecord[self.cnt] = POI
        self.POISum += POI
        self.POI2Sum += POI*POI
        self.otherSum += other
        self.cnt += 1
        if self.cnt == self.granularity:
            self.POIBins[self.curr_bin] = self.POISum
            self.POI2Bins[self.curr_bin] = self.POI2Sum
            self.otherBins[self.curr_bin] = self.otherSum
            corr = 0
            lastCorr = 0.5
            done = False

            if DEBUG > 1:
                print()
                print("BINS:")
                print(self.POIBins)
                print(self.POI2Bins)
                if (self.POI2Bins[self.curr_bin] - self.POI2Bins[self.curr_bin - 1]) != 0:
                    print(1)

            while(not done):
                corr = np.sum(np.roll(self.POITmpRecord, self.curr_corr_len)[self.curr_corr_len:]
                              * self.POITmpRecord[self.curr_corr_len:]) / (self.granularity - self.curr_corr_len)
                corr -= np.square((self.POIBins[self.curr_bin] - self.POIBins[self.curr_bin-1])/self.granularity)
                var = (self.POI2Bins[self.curr_bin] - self.POI2Bins[self.curr_bin - 1])/self.granularity\
                      - np.square((self.POIBins[self.curr_bin] - self.POIBins[self.curr_bin-1])/self.granularity)
                if var > 0:
                    corr /= var
                else:
                    corr = 1
                done = True
                if corr < 0: 
                    corr = 0
                if corr < 0.25 and lastCorr <= 0.75 and self.curr_corr_len >= 2:
                    self.curr_corr_len = int(self.curr_corr_len / 2)
                    done = False
                if corr > 0.75 and lastCorr >= 0.25 and self.curr_corr_len < int(self.granularity / 4):
                    self.curr_corr_len = int(self.curr_corr_len * 2)
                    done = False
                lastCorr = corr

            corr = np.power(corr, 1/self.curr_corr_len)
            self.corrBins[self.curr_bin] = corr

            b = int(self.curr_bin / 2)
            while(b >= 1):
                block1 = (self.POIBins[self.curr_bin] - self.POIBins[self.curr_bin - b]) / (self.granularity * b)
                block2 = (self.POIBins[self.curr_bin - b] - self.POIBins[self.curr_bin - 2 * b]) / (self.granularity * b)
                if abs(block1 - block2) < self.prec:
                    bin1 = self.curr_bin - 2 * b
                    n = self.granularity * 2 * b
                    avg_corr = np.average(self.corrBins[bin1+1:self.curr_bin+1]) # It's good to use np.average here due to the +1 natural
                    # avg_corr = corr
                    var = ((self.POI2Bins[self.curr_bin] - self.POI2Bins[bin1]) / n \
                          - np.square((self.POIBins[self.curr_bin] - self.POIBins[bin1]) / n)) / n
                    if (1 - avg_corr) != 0:
                        # var is small and corr approch to 0
                        var *= (1 + avg_corr) / (1 - avg_corr)
                        if DEBUG == 1:
                            print("var =", var)
                        if var < self.prec**2 and var > 0:
                            self.finalPOI = (self.POIBins[self.curr_bin] - self.POIBins[bin1]) / n
                            self.finalOther = (self.otherBins[self.curr_bin] - self.otherBins[bin1]) / n
                            self.converged = True
                            self.equil = n
                            self.steps = 2 * b
                            break
                    elif self.curr_bin == self.patience:
                        print("OUT OF PATIENCE!!!!!!!!!!!!!!!!!!!!!")
                        self.finalPOI = (self.POIBins[self.curr_bin] - self.POIBins[bin1]) / n
                        self.finalOther = (self.otherBins[self.curr_bin] - self.otherBins[bin1]) / n
                        self.converged = True
                        self.equil = n
                        self.steps = 2 * b
                        break
                b -= 1

            self.curr_bin += 1
            self.cnt = 0
            # again, following condition test is valid due to the +1 natural
            if self.curr_bin == self.POIBins.shape[0]:
                self.POIBins[np.arange(self.nb_bins / 2, dtype=int) + 1] = self.POIBins[
                    np.arange(self.nb_bins / 2, dtype=int) * 2 + 2]
                self.POIBins[int(self.nb_bins / 2) + 1:] = 0

                self.POI2Bins[np.arange(self.nb_bins / 2, dtype=int) + 1] = self.POI2Bins[
                    np.arange(self.nb_bins / 2, dtype=int) * 2 + 2]
                self.POI2Bins[int(self.nb_bins / 2) + 1:] = 0

                self.otherBins[np.arange(self.nb_bins / 2, dtype=int) + 1] = self.otherBins[
                    np.arange(self.nb_bins / 2, dtype=int) * 2 + 2]
                self.otherBins[int(self.nb_bins / 2) + 1:] = 0

                self.corrBins[np.arange(self.nb_bins / 2, dtype=int) + 1] = (self.corrBins[
                                                                                 np.arange(self.nb_bins / 2,
                                                                                           dtype=int) * 2 + 2] +
                                                                             self.corrBins[
                                                                                 np.arange(self.nb_bins / 2,
                                                                                           dtype=int) * 2 + 1]) / 2
                self.corrBins[int(self.nb_bins / 2) + 1:] = 0

                self.curr_bin = int(self.nb_bins / 2 + 1)
                self.granularity = int(self.granularity * 2)
                if DEBUG > 0:
                    print("granularity resized:", self.granularity)
                self.POITmpRecord = np.zeros(self.granularity)

        return self.converged

class ugly_equilibrium_detector:
    def __init__(self, granularity, nb_bins, prec):
        assert not nb_bins%2, "nb_bins must be a even number"
        self.nb_bins = nb_bins
        self.prec = prec
        # POI: Property of Interest
        self.POISum = 0
        self.POI2Sum = 0
        self.POITmpRecord = np.zeros(granularity)
        # +1 because curr_bin initialize to be 1, be careful when using np.average, due to the +1
        self.POIBins = np.zeros(nb_bins+1) # store POISum
        self.POI2Bins = np.zeros(nb_bins+1)  # store POI^2Sum
        self.corrBins = np.zeros(nb_bins+1)
        self.otherSum = 0
        self.otherBins = np.zeros(nb_bins+1)

        self.finalPOI = np.inf
        self.finalOther = np.inf

        self.granularity = granularity
        self.curr_corr_len = int(granularity / 2)
        self.cnt = 0
        # cur_bin start from 1 since we always leave 0th bin to be 0, since that's with bin with nothing sumed
        self.curr_bin = 1

        self.converged = False
        self.equil = 0
        self.steps = 0


    def reset(self):
        self.__init__(granularity=self.granularity, nb_bins=self.nb_bins, prec=self.prec)

    def get_POI(self):
        return self.finalPOI

    def get_other(self):
        return self.finalOther

    def new_data(self, POI, other):
        self.POITmpRecord[self.cnt] = POI
        self.POISum += POI
        self.POI2Sum += POI*POI
        self.otherSum += other
        self.cnt += 1
        if self.cnt == self.granularity:
            self.POIBins[self.curr_bin] = self.POISum
            self.POI2Bins[self.curr_bin] = self.POI2Sum
            self.otherBins[self.curr_bin] = self.otherSum

            b = int(self.curr_bin / 2)
            while(b >= 1):
                block1 = (self.POIBins[self.curr_bin] - self.POIBins[self.curr_bin - b]) / (self.granularity * b)
                block2 = (self.POIBins[self.curr_bin - b] - self.POIBins[self.curr_bin - 2 * b]) / (self.granularity * b)
                if abs(block1 - block2) < self.prec:
                    bin1 = self.curr_bin - 2 * b
                    n = self.granularity * 2 * b
                    # TODO: calculate length needed from var
                    # var = (self.POI2Bins[self.curr_bin] - self.POI2Bins[bin1]) / n \
                    #       - np.square((self.POIBins[self.curr_bin] - self.POIBins[bin1]) / n)
                    self.finalPOI = (self.POIBins[self.curr_bin] - self.POIBins[bin1]) / n
                    self.finalOther = (self.otherBins[self.curr_bin] - self.otherBins[bin1]) / n
                    self.converged = True
                    self.equil = n
                    self.steps = 2 * b
                    break
                b -= 1

            self.curr_bin += 1
            self.cnt = 0
            # again, following condition test is valid due to the +1 natural
            if self.curr_bin == self.POIBins.shape[0]:
                self.POIBins[np.arange(self.nb_bins / 2, dtype=int) + 1] = self.POIBins[
                    np.arange(self.nb_bins / 2, dtype=int) * 2 + 2]
                self.POIBins[int(self.nb_bins / 2) + 1:] = 0

                self.POI2Bins[np.arange(self.nb_bins / 2, dtype=int) + 1] = self.POI2Bins[
                    np.arange(self.nb_bins / 2, dtype=int) * 2 + 2]
                self.POI2Bins[int(self.nb_bins / 2) + 1:] = 0

                self.otherBins[np.arange(self.nb_bins / 2, dtype=int) + 1] = self.otherBins[
                    np.arange(self.nb_bins / 2, dtype=int) * 2 + 2]
                self.otherBins[int(self.nb_bins / 2) + 1:] = 0

                self.curr_bin = int(self.nb_bins / 2 + 1)
                self.granularity = int(self.granularity * 2)
                self.POITmpRecord = np.zeros(self.granularity)

        return self.converged

class ugly_equilibrium_detector2:
    def __init__(self, granularity, nb_bins, prec):
        assert not nb_bins%2, "nb_bins must be a even number"
        self.nb_bins = nb_bins
        self.prec = prec
        # POI: Property of Interest
        self.POISum = 0
        self.POI2Sum = 0
        self.POITmpRecord = np.zeros(granularity)
        # +1 because curr_bin initialize to be 1, be careful when using np.average, due to the +1
        self.POIBins = [] # store POISum
        self.POIBins.append(0)
        self.POI2Bins = []  # store POI^2Sum
        self.POI2Bins.append(0)
        self.otherSum = 0
        self.otherBins = []
        self.otherBins.append(0)

        self.finalPOI = np.inf
        self.finalOther = np.inf

        self.granularity = granularity
        self.curr_corr_len = int(granularity / 2)
        self.cnt = 0
        # cur_bin start from 1 since we always leave 0th bin to be 0, since that's with bin with nothing sumed
        self.curr_bin = 1

        self.converged = False
        self.equil = 0
        self.steps = 0

    def get_POI(self):
        return self.finalPOI

    def get_other(self):
        return self.finalOther

    def new_data(self, POI, other):
        self.POITmpRecord[self.cnt] = POI
        self.POISum += POI
        self.POI2Sum += POI*POI
        self.otherSum += other
        self.cnt += 1
        if self.cnt == self.granularity:
            self.POIBins.append(self.POISum)
            self.POI2Bins.append(self.POI2Sum)
            self.otherBins.append(self.otherSum)

            b = int(self.curr_bin / 2)
            while(b >= 1):
                block1 = (self.POIBins[self.curr_bin] - self.POIBins[self.curr_bin - b]) / (self.granularity * b)
                block2 = (self.POIBins[self.curr_bin - b] - self.POIBins[self.curr_bin - 2 * b]) / (self.granularity * b)
                if abs(block1 - block2) < self.prec:
                    bin1 = self.curr_bin - 2 * b
                    n = self.granularity * 2 * b
                    # TODO: calculate length needed from var
                    # var = (self.POI2Bins[self.curr_bin] - self.POI2Bins[bin1]) / n \
                    #       - np.square((self.POIBins[self.curr_bin] - self.POIBins[bin1]) / n)
                    self.finalPOI = (self.POIBins[self.curr_bin] - self.POIBins[bin1]) / n
                    self.finalOther = (self.otherBins[self.curr_bin] - self.otherBins[bin1]) / n
                    self.converged = True
                    self.equil = n
                    self.steps = 2 * b
                    break
                b -= 1

            self.curr_bin += 1
            self.cnt = 0

        return self.converged

class ema_equilibrium_detector:
    """
    nb: number of consective ema < prec delta needed to trigger convergence
    """
    def __init__(self, prec, nb):
        self.his = []
        self.nb = nb
        self.prec = prec
        # POI: Property of Interest
        self.nPointToMenmorize = np.linspace(2, 1000, 999)
        self.beta = (self.nPointToMenmorize - 1) / self.nPointToMenmorize
        self.POIEma = np.zeros(self.beta.shape)
        self.emaRecord = np.full((self.nb, self.beta.shape[0]), np.inf)
        self.otherEma = np.zeros(self.beta.shape)
        self.finalIdx = 0
        self.cnt = 0
        self.equil = 0
        self.steps = None

    def get_POI(self):
        return self.POIEma[self.finalIdx]

    def get_other(self):
        return self.otherEma[self.finalIdx]

    def get_his(self, idx = None):
        if idx == None:
            return np.array(self.his)[:, self.finalIdx]
        else:
            return np.array(self.his)[:, idx]

    def new_data(self, POI, other):
        self.cnt += 1
        prevPOIEma = self.POIEma[self.nPointToMenmorize <= self.cnt]
        self.POIEma = self.POIEma * self.beta + POI * (1 - self.beta)
        self.his.append(self.POIEma)
        self.otherEma = self.otherEma * self.beta + other * (1 - self.beta)
        delta = np.abs(prevPOIEma - self.POIEma[self.nPointToMenmorize <= self.cnt])
        delta = np.pad(delta, (0, self.beta.shape[0] - delta.shape[0]), 'constant', constant_values=(0, np.inf))
        self.emaRecord[self.cnt%self.nb] = delta
        self.finalIdx = np.argmin(np.max(self.emaRecord, axis=0))
        if self.cnt > self.nb and delta[self.finalIdx] < self.prec:
            self.equil = self.nPointToMenmorize[self.finalIdx]
            return True
        return False


# data = np.full((50000), 2) + np.random.rand(50000)
# data = np.append(np.arange(3), np.full((2000), 3))
#
# X = np.linspace(0.5, 1000, 5000)
# data = 1/X+1 + np.random.normal(0, 1, 5000)*0.1
#
#
#
# X = np.linspace(0.5, 1000, 5000)
# data = np.sin(X)
# plt.plot(data)
# plt.show()

# err = []
# for runs in range(1):
#     print("test start!")
#     test_eq = ema_equilibrium_detector(prec=1e-6, nb=700)
#     # test_eq = ugly_equilibrium_detector(granularity=500, prec=1e-2, nb_bins=qq)
#
#     a = 1500
#     b = 100000
#     X = np.linspace(0, 2000, a + b)
#     # data = np.append(np.linspace(0, 1000, a), np.full((b), 1000)) + np.random.normal(0, 10, a + b) + np.sin(X*0.2)*5
#     data = np.append(np.linspace(0, 1000, a), np.full((b), 1000)) + np.random.normal(0, 5, a + b)
#
#     if runs == 0:
#         plt.plot(data)
#         plt.show()
#
#     for idx, d in enumerate(data):
#         if test_eq.new_data(POI=d, other=1):
#             POI = test_eq.get_POI()
#             other = test_eq.get_other()
#             print("converged with:", POI, other, "with error:", np.abs(1000 - POI))
#             print("after", idx, "samples, averaged over", test_eq.equil, "samples, with 2 * b =", test_eq.steps)
#             plt.plot(data)
#             plt.plot(test_eq.get_his())
#             plt.plot(test_eq.get_his(1))
#             plt.plot(test_eq.get_his(200))
#             plt.legend(["data", "best", "1", "20"])
#             plt.show()
#             err.append(np.abs(1000 - POI))
#             break
# print()
# print(np.average(err))
# exit()


BP = []
def disconti_detector(ys, newY, v_ys_newY = None, small_value=0.001, numPred = 1, maxYLength = 20, mu=None):
    """
    mu is just for DEBUG purpose!!!

    Take maxYLength number of previous ys, find best order of fit, fit a model, make numPred number
    of predictions with uncertanties, compare that with newY with uncertainty, return the z test value.

    If uncertainty was provided, provided uncertainty was used, otherwise, calculate untertainty from
    fitted model.

    DEBUG == 0: print nothing
    DEBUG == 1: print intermedit values
    DEBUG == 2: plot everything
    """
    yLength = len(ys)
    if yLength < 3:
        BP.append(-1)
        return 0
    if yLength > maxYLength: return disconti_detector(ys[-maxYLength:], newY, v_ys_newY, small_value, maxYLength=maxYLength, mu=mu)

    # create arbitury x for fittings and etc
    x = np.arange(len(ys))

    # Calculate CV score, finding best order
    bestP = 0
    bestCV = np.inf
    bestBIC = np.inf
    BICCounter = 0
    for p in range(yLength - 1):
        # Cross Validation
        # CV = find_CV(x, ys, p)
        # if CV + 1e-15 < bestCV:
        #     bestCV = CV
        #     bestP = p
        # BIC
        fit = np.polyfit(x, ys, p)
        fn = np.poly1d(fit)
        SSE = np.sum(np.square(fn(x) - ys))
        BIC = yLength*np.log(SSE/yLength) + np.log(yLength)*p
        BICCounter += 1
        if BIC + 1e-15 < bestBIC:
            bestBIC = BIC
            bestP = p
            BICCounter = 0
        if BICCounter > 7: break

    # bestP = 3
    BP.append(bestP)
    # fit poly using the best order
    fitResult = np.polyfit(x, ys, bestP)
    fn = np.poly1d(fitResult)

    if DEBUG:
        print(fn)
        print("bestP : ", bestP)
        print("newYbar : ", fn(x[-1]+np.arange(numPred)+1))

    res = np.sum(np.square(fn(x) - ys))/yLength
    bigX = np.power(x.reshape(yLength, 1), np.arange(bestP+1))
    V = res * np.linalg.inv( np.matmul(bigX.T, bigX) )

    newX = np.power(len(ys)+np.arange(numPred).reshape(numPred, 1), np.arange(bestP+1))
    tmp = np.matmul(newX, V)
    littlev = np.matmul(tmp, newX.T)

    if DEBUG: print("little v: ",littlev)

    if v_ys_newY is not None:
        sigma2 = np.average(v_ys_newY)
    else:
        sigma2 = np.sum(np.square(fn(x) - ys))/(yLength-1)

    if DEBUG > 1:
        plt.errorbar(x, ys, ls='dotted', yerr=np.full(len(ys), SCALE*2))
        plt.plot(x, fn(x))
        plt.errorbar(x[-1]+np.arange(numPred)+1, fn(x[-1]+np.arange(numPred)+1), yerr=np.sqrt(littlev.diagonal())*2, capsize=5)
        plt.errorbar(x[-1]+1, newY, yerr=np.sqrt(sigma2 + np.sqrt(small_value)), capsize=5)
        plt.show()

    if DEBUG:
        print("sigma: ", np.sqrt(sigma2))
        print("at mu=", mu, "    z score: ", abs(fn(x[-1]+np.arange(numPred)+1) - newY)/np.sqrt(littlev + sigma2 + np.sqrt(small_value)))

    return abs(fn(x[-1]+np.arange(numPred)+1) - newY)/np.sqrt(littlev + sigma2 + np.sqrt(small_value))

def get_spin_change(n_sites, steps):
    """
    get total number of spin changes that resulting a different state than init
    """
    moves = 0
    sites = np.full((n_sites), False)
    for step in steps:
        sites[step.last_move[0].index] = ~sites[step.last_move[0].index]
        moves += np.sum(sites)

    return moves


# deltSpinDetecter = np.array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  4, 14, 21, 24, 25, 26, 14, 12,  3,  0,  0,  0,  0,  0,  0,  0,  0,])
# deltSpinDetecter = np.flip(deltSpinDetecter)

# critialIdx = np.inf
# for critialIdx, ds in enumerate(deltSpinDetecter):
#     if disconti_detector(ys=deltSpinDetecter[:critialIdx], newY=deltSpinDetecter[critialIdx], maxYLength=20) > Z:
#         break
# critialIdx = -1*(critialIdx+1)

# print(critialIdx)

# exit(0)
chgnet = CHGNet.load()

GroundStates = options["GroundStates"]
gs_db_names = options["EMC"]["gs_db_names"]
muInit, muFinal = options["EMC"]["muInit"], options["EMC"]["muFinal"]
dMu = options["EMC"]["dMu"]

dMu = abs(dMu)
if (muFinal - muInit)*dMu < 0:
    dMu = -dMu
# print(mus)
# exit()
gsE = [0,0]
for gsIdx, gs_name in enumerate(GroundStates):
        # if gsIdx == 0:
    #     continue

    # gs_db_name = "LiNa/LiNa-64-0.0-EIM.db"
    # gs_db_name = "Li-gs.db"
    db_name = options["CLEASE"][gs_name]["CESettings"]["db_name"]
    conc = Concentration(basis_elements=options["CLEASE"][gs_name]["CESettings"]["concentration"])

    tmp = basis_elements=options["CLEASE"][gs_name]["CESettings"].copy()
    tmp['concentration'] = conc
    MCsettings = CEBulk(**tmp)

    eciName = options["CLEASE"][gs_name]["CEFitting"]["ECI_filename"]
    if eciName == "FROM DB":
        with open(db_name + "-eci.json") as f:
            eci = json.load(f)
    else:
        with open(eciName) as f:
            eci = json.load(f)

    db = connect(gs_db_names[gsIdx])
    gs05 = None
    for row in db.select(""):
        gs05 = row.toatoms()

    # gs05 = attach_calculator(MCsettings, atoms=gs05, eci=eci)
    # gs05.calc = EAM(potential='AuCu_Zhou04.eam.alloy')
    gs05.calc = ASElammps
    
    # gs05.calc = CHGNetCalculator(model=chgnet)
    
    gsE[gsIdx] = gs05.get_potential_energy()/len(gs05)
    

for gsIdx, gs_name in enumerate(GroundStates):
    # if gsIdx == 0:
    #     continue

    # gs_db_name = "LiNa/LiNa-64-0.0-EIM.db"
    # gs_db_name = "Li-gs.db"
    db_name = options["CLEASE"][gs_name]["CESettings"]["db_name"]
    conc = Concentration(basis_elements=options["CLEASE"][gs_name]["CESettings"]["concentration"])

    tmp = basis_elements=options["CLEASE"][gs_name]["CESettings"].copy()
    tmp['concentration'] = conc
    MCsettings = CEBulk(**tmp)

    eciName = options["CLEASE"][gs_name]["CEFitting"]["ECI_filename"]
    if eciName == "FROM DB":
        with open(db_name + "-eci.json") as f:
            eci = json.load(f)
    else:
        with open(eciName) as f:
            eci = json.load(f)

    # atoms = MCsettings.prim_cell.copy()
    # atoms *= (4,4,4)
    # atoms.write(options["EMC"]["gs_db_names"][0])

    # atoms = MCsettings.prim_cell.copy()
    # atoms.numbers[0] = atomic_numbers[options["CLEASE"][gs_name]["CESettings"]["concentration"][0][1]]
    # atoms *= (4,4,4)
    # atoms.write(options["EMC"]["gs_db_names"][1])

    db = connect(gs_db_names[gsIdx])
    gs05 = None
    for row in db.select(""):
        gs05 = row.toatoms()

    gs05 = attach_calculator(MCsettings, atoms=gs05, eci=eci)
    # gs05.calc = EAM(potential='AuCu_Zhou04.eam.alloy')
    # gs05.calc = ASElammps
    
    # gs05.calc = CHGNetCalculator(model=chgnet)

    gs05.info["gsE"] = gsE
    # view(gs05)


    # ts = time.time()
    # for i in range(int(1e8)):
    #     # print("gs atom E:", gs05.get_potential_energy())
    #     if gs05.numbers[0] == 3:
    #         gs05.numbers[0] = 12
    #     else:
    #         gs05.numbers[0] = 3
    # te = time.time()
    # print(te-ts)

    # exit(0)

    # NP = 20
    #
    # np.random.seed(int(time.time()))
    # x = np.linspace(0, 5, NP)
    # y = -10*(x-4)**3 - 50*(x-4)**2 + 5 + np.random.normal(scale=SCALE, size=NP)
    # # y = [1, -1, 1, -2, 4]
    # print("z test score: ", disconti_detector(y, -125, maxYLength=np.inf))
    # exit(0)

    # Ts = np.linspace(150, 700, 1)
    # mus = np.linspace(0.1209, 0.45, 50)
    # mus = np.linspace(0.34, 0.45, 1)
    # print("Ts: ", Ts)

    old_spins = np.copy(gs05.numbers)

    deltSpin = []
    deltSpin2 = []
    deltSpin2Detecter = []

    X4C = []
    C4C = []
    C = []

    # if j_tmp == 0 and breaked, it means the outter variable also reaches phase boundray, and toExit = True
    toExit = False
    breakingPoint = []

    avgWindowWidth = 100

    TInit, TFinal = options["EMC"]["TInit"], options["EMC"]["TFinal"]
    dT = options["EMC"]["dT"]
    Ts = np.arange(TInit, TFinal+1e-5, dT)
    # Ts = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
    if gsIdx == 0:
        mus = np.arange(muInit, muFinal+dMu*1e-5, dMu)
    elif gsIdx == 1:
        muTmp = muInit
        muInit = muFinal
        muFinal = muTmp
        dMu = -dMu
        mus = np.arange(muInit, muFinal+dMu*1e-5, dMu)

    # print(gs_name, mus[0])
    # exit(0)



    # TInit, TFinal = 3.2e5, 4e5
    # dT = 5e3 * 1e8
    # Ts = np.arange(TInit, TFinal+1e-5, dT)
    # if gs_db_name == "1.0.db":
    #     muInit, muFinal =  30, -35
    #     dMu = -1#  * 1e8
    #     mus = np.arange(muInit, muFinal+dMu*1e-5, dMu)
    # else:
    #     muInit, muFinal =  -35, 30
    #     dMu = 1#  * 1e8
    #     mus = np.arange(muInit, muFinal+dMu*1e-5, dMu)  
        
    if gs_name == 'L':
        phi_lte = -0.0516199
    elif gs_name == 'M':
        phi_lte = -0.1003174266
    else:
        lte = LTE()
        lte.set_gs_atom(gs05)
        phi_lte = lte.get_E(T=Ts[0], mu=mus[0])
    # phi_lte = gs05.get_potential_energy()
    # phi_lte = - mus[0] * get_conc(gs05)

    print("######################################")
    print("at T =", Ts[0], gs_name, "E_lte = ", phi_lte)
    print("######################################")
    # continue


    # from IsingCalculator import IsingCalc
    # calc = IsingCalc(E0=0, h=1, J=5, cutoff=3)

    # mf = MF()
    # mf.set_gs_atom(gs05, IsingCalc=calc)
    # phi_mf = mf.get_phi(T=Ts[0], x = 1, mu=mus[0])
    # print("phi_mf =", phi_mf)

    # print("diff :", phi_mf - phi_lte)

    # exit()

    SCALE = 1

    TFirst = False
    # increasing = [Temp, Mu]
    increasing = np.array([1, 0], dtype=bool)
    if not TFirst:
        increasing = ~increasing

    prevGs05 = gs05.copy()
    print("hhhhhhh", gs05.get_potential_energy())

    iRange = np.sum([len(Ts), len(mus)]*~increasing)
    jRange = np.sum([len(Ts), len(mus)]*increasing)
    phiTable = np.zeros((iRange, jRange))
    print(phiTable.shape)
    ETable = np.zeros((iRange, jRange))
    XTable = np.zeros((iRange, jRange))
    # muTable = np.zeros((iRange, jRange))
    deltSpinDetecter = np.zeros((iRange, jRange))
    phiTable[0, 0] = phi_lte
    systemSize = len(gs05.numbers)
    timeStarted = time.time()
    
    # myTraj = []
    
    for i_tmp in range(iRange):
        for j_tmp in range(jRange):
            currTi = np.sum([i_tmp, j_tmp] * ~increasing)
            currMui = np.sum([i_tmp, j_tmp] * increasing)
            temp = Ts[currTi]
            mu = mus[currMui]
            print("T = ", temp, "mu = ", mu)
            mcSGC = SGCMonteCarlo(gs05, temp, symbols=['Mg', 'Li'])
            # Using move observer
            # obs = MoveObserver(gs05, only_accept=True)
            # mcSGC.attach(obs, interval=1)
            E = []
            singE = []
            numOfAu = []
            MaxIterNum = 1000000
            avgRecorder = []

            ################# ！！！！！！！！！！！！！！！！ ###################
            eq = equilibrium_detector(granularity=6, nb_bins=16, prec=options["EMC"]["error"], patience=6)
            # eq = ema_equilibrium_detector(prec=2e-6, nb=5000)

            """uncomment below for real runs"""
            # run until converge
            for i in range(MaxIterNum):
                if not i % 100000:
                    print(i)
                mcSGC.run(steps=100, chem_pot={'c1_0': mu})
                # myTraj.append(gs05.copy())
                currE = mcSGC.get_thermodynamic_quantities()['sgc_energy']
                # singE.append(mcSGC.get_thermodynamic_quantities()['singlet_energy'])
                # currE = mcSGC.get_thermodynamic_quantities()['singlet_energy']
                # print(gs05.get_potential_energy(), currE, currSCGE)
                E.append(currE)
                currX = get_conc(gs05)
                numOfAu.append(currX)

                if not i % avgWindowWidth:
                    avgRecorder.append(np.average(E[-100:]))

                # if converged, save qautities of interest, get ready to integrate
                if eq.new_data(POI=currX, other=currE) or i == MaxIterNum-1:
                # if i > 2000 or i == MaxIterNum - 1:
                    print("converged after: ", i)
                    ###############################################
                    # print("mu = ", mu, "x = ", eq.get_POI())
                    # exit(0)
                    ###############################################
                    # save converged state of [i_tmp, 0] for [i_tmp+1, 0]
                    if j_tmp == 0:
                        # prevGs05 = gs05.copy()
                        # view(prevGs05)

                        # ！！！！！！！！！！！！！！！！this only works with µ first！！！！！！！！！！！！！
                        db = connect(gs_db_names[gsIdx])
                        for row in db.select(""):
                            prevGs05 = row.toatoms()
                    if DEBUG > 1:
                        plot_stuff(range(i+1), [E, numOfAu, eq.get_his()], myTitle=[str(temp)])
                        plt.show()
                        view(gs05)
                    if DEBUG:
                        print(gs05)
                        print("E", eq.get_other(), "x", eq.get_POI())
                    # saving qautities of interest, if is to be able to collapse code
                    if 1:
                        ETable[i_tmp, j_tmp] = eq.get_other()/systemSize
                        XTable[i_tmp, j_tmp] = eq.get_POI()
                        # muTable[i_tmp, j_tmp] = eci2chem_pot(bf_ls, gs05.calc.eci)
                        deltSpinDetecter[i_tmp][j_tmp] = np.sum(old_spins != gs05.numbers)
                        old_spins = np.copy(gs05.numbers)
                    
                    if j_tmp == 0 and np.sum(old_spins != gs05.numbers) > 1.0:
                        assert False, "Please adjust mu"

                    # reset chemical potential
                    tmpNum = gs05.numbers
                    mcSGC.run(steps=0, chem_pot={'c1_0': 0})
                    for (a, b) in zip(tmpNum, gs05.numbers):
                        assert a == b
                    # exit()
                    break
            
            # write(gs_name + str(mu) + ".xyz", myTraj)
        
            if DEBUG:
                if j_tmp:
                    print(deltSpinDetecter[i_tmp, :j_tmp+1])
                else:
                    print(deltSpinDetecter[:i_tmp+1, 0])

            # if this is the first point, no need to bother
            if i_tmp == 0 and j_tmp == 0:
                continue
            
            #!!!!!!!!!!!!!!!!!!!!!!!!! TMP FIX !!!!!!!!!!!!!!!!!!!!!!
            if j_tmp == 0:
                lte.set_gs_atom(attach_calculator(MCsettings, atoms=gs05.copy(), eci=eci))
                phiTable[i_tmp, j_tmp] = lte.get_E(T=temp, mu=mu)
                continue

            # if j_tmp:
            #     myZScore = disconti_detector(ys=deltSpinDetecter[i_tmp][:j_tmp], newY=deltSpinDetecter[i_tmp][j_tmp],
            #                                  maxYLength=20, mu=mu)
            # else:
            #     myZScore = disconti_detector(ys=deltSpinDetecter[:i_tmp][0], newY=deltSpinDetecter[i_tmp][0],
            #                                  maxYLength=20, mu=mu)
            #
            # # if not breaked and myZScore > ZTestAcc:
            # if myZScore > Z:
            #     # i_tmp 方向出现discontinuity，end everything
            #     if j_tmp == 0:
            #         toExit = True
            #     break

            """below is loops-testing code"""
            # ETable[i_tmp, j_tmp] = temp
            # XTable[i_tmp, j_tmp] = mu
            # breakOrNot = bool(int(input()))
            # if breakOrNot:
            #     if j_tmp == 0:
            #         toExit = True
            #     break

            # passed discontiouity test, proceed to integration
            # TODO finish this and work on avergeing length and GPR
            actual_mu = mu #- 0.13660105109875623
            # if increasing T first
            # NAtoms = len(gs05.numbers)
            if increasing[0]:
                if j_tmp==0:
                    phiTable[i_tmp, j_tmp] = phiTable[i_tmp - 1, j_tmp] \
                                            - (XTable[i_tmp, j_tmp] + XTable[i_tmp - 1, j_tmp]) / 2 * dMu
                else:
                    prevB = 1/kB/(temp - dT)
                    currB = 1/kB/temp
                    phiTable[i_tmp, j_tmp] = (phiTable[i_tmp, j_tmp-1]*prevB
                                            + (ETable[i_tmp, j_tmp] - actual_mu * XTable[i_tmp, j_tmp]
                                                + ETable[i_tmp, j_tmp-1] - actual_mu * XTable[i_tmp, j_tmp-1])
                                            * (currB - prevB) / 2) / currB
            else:
                # if j_tmp == 0, we need to integrate from a previous T of the same µ
                # if j_tmp == 0:
                    # prevB = 1 / kB / (temp - dT)
                    # currB = 1 / kB / temp
                    # # phiTable[i_tmp, j_tmp] = (phiTable[i_tmp - 1, j_tmp] * prevB
                    # #                           + (ETable[i_tmp, j_tmp] - actual_mu * XTable[i_tmp, j_tmp]
                    # #                              + ETable[i_tmp - 1, j_tmp] - actual_mu * XTable[i_tmp - 1, j_tmp])
                    # #                           * (currB - prevB) / 2) / currB

                    # aa = phiTable[i_tmp - 1, j_tmp] * prevB / currB
                    # # bb = ETable[i_tmp, j_tmp] - actual_mu * XTable[i_tmp, j_tmp]
                    # # cc = ETable[i_tmp - 1, j_tmp] - actual_mu * XTable[i_tmp - 1, j_tmp]
                    # # dd = (bb + cc) *  (currB - prevB) / 2
                    # dd = (ETable[i_tmp, j_tmp] + ETable[i_tmp - 1, j_tmp]) * (currB - prevB) / 2
                    # # if gs_name == "Mg":
                    # #     dd = (-1) * (currB - prevB) / 2

                    # ee = dd / currB

                    # phiTable[i_tmp, j_tmp] = aa + ee
                if j_tmp == 0:
                    lte.set_gs_atom(gs05)
                    phiTable[i_tmp, j_tmp] = lte.get_E(T=temp, mu=actual_mu)
                    # mf.set_gs_atom(gs05, IsingCalc=calc)
                    # phiTable[i_tmp, j_tmp] = mf.get_phi(T=temp, x = int(gs_db_name[0]), mu=actual_mu)
                    # phiTable[i_tmp, j_tmp] = gs05.get_potential_energy()
                else:
                    phiTable[i_tmp, j_tmp] = phiTable[i_tmp, j_tmp-1] \
                                            - (XTable[i_tmp, j_tmp] + XTable[i_tmp, j_tmp - 1]) / 2 * dMu

        # Inner loop finished or breaked, setup for next inner scan:
        if toExit:
            break
        gs05 = attach_calculator(MCsettings, atoms=prevGs05.copy(), eci=eci)
        # gs05 = prevGs05.copy()

        old_spins = np.copy(gs05.numbers)


    # deltSpinDetecter = np.flip(XTable, axis=1)
    # print(deltSpinDetecter)

    # critialIdx = np.inf
    # for critialIdx, ds in enumerate(deltSpinDetecter[0]):
    #     if ds < 1- 1e-3:
    #         break
    #     # if disconti_detector(ys=deltSpinDetecter[0][:critialIdx], newY=deltSpinDetecter[0][critialIdx], maxYLength=20) > Z:
    #     #     break
    # critialIdx = -1*(critialIdx+1)
    # print(critialIdx)

    # XTable[0,0] = critialIdx

    # np.save("aucu-XTable-2550-" + gs_db_name[0:4] + ".npy", XTable)
    # np.save("aucu-phiTable-2550-" + gs_db_name[0:4] + ".npy", phiTable)
    # np.save("ref-ETable-" + gs_db_name[0:3] + ".npy", ETable)
    # np.save("chemPot-h1j5-XTable-" + gs_db_name[0:3] + ".npy", XTable)


    np.save("300-900-XTable-" + gs_name + ".npy", XTable)
    # np.save("lina-muTable-" + gs_db_name[-10:-7] + ".npy", muTable)
    np.save("300-900-phiTable-" + gs_name + ".npy", phiTable)

    plt.figure(1)
    plt.plot(mus, phiTable[-1], '-o')
    plt.figure(2)
    plt.plot(mus, XTable[-1], '-o')


# exit(0)
timeEnded = time.time()

print("walltime:", timeEnded - timeStarted)


# print(deltSpinDetecter)


# plt.plot(mus, XTable[0], '-o')
# plt.vlines(mus[critialIdx], 0, 1)
plt.figure(1)
plt.title("Plot to set mu")
plt.xlabel("mu")
plt.ylabel("phi")
plt.savefig('mu_phi.png')

plt.figure(2)
plt.xlabel("mu")
plt.ylabel("x")
plt.savefig('mu_x.png')
# plt.ylim([0, 1])
# plt.figure()
# plt.plot(muTable[0], XTable[0], '-o')
# plt.title("Actual mu vs X")
# plt.show()
# plt.plot(phiTable[0])

# np.save("toy-phiTable-1.0-left.npy", phiTable)
# np.save("toy-XTable-1.0-left.npy", XTable)
# np.save("toy-ETable-1.0-left.npy", ETable)

# exit()



# plot_stuff(mus, [avgEs, avgXs, deltSpin, deltSpin2], ["E"+str(temp), "X"+str(temp), "delta spin"+str(temp), "delta spin2"+str(temp)], breakingpt=breakingPoint)
# plt.show()

# np.save("X4C.npy", np.array(X4C))
# np.save("C4C.npy", np.array(C4C))
# print(gs05)
# print(E[-1])
# view(gs05)
# gs05.write("0.5.2.db")
# plt.figure()
# plt.plot(numOfAu)
# plt.figure()


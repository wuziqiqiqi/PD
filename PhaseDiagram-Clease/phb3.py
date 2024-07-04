import json
import logging
import sys, os, time

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from lammps import lammps

from ase.db import connect
from ase.calculators.emt import EMT
from ase.calculators.eam import EAM
from ase.calculators.lammpslib import LAMMPSlib
from ase.visualize import view
from ase import Atom, Atoms
from ase.build import bulk

from clease.settings import Concentration
from clease.structgen import NewStructures
from clease.settings import CEBulk
from clease.calculator import attach_calculator
from clease.montecarlo import Montecarlo, constraints, SGCMonteCarlo, BinnedBiasPotential, MetaDynamicsSampler
from clease.montecarlo.observers import ConcentrationObserver, MoveObserver, CorrelationFunctionObserver
from clease.tools import update_db
from clease import Evaluate
import clease.plot_post_process as pp

from drawpd import LTE, MF

kB = 8.617333262e-5

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
DEBUG = 1


ZTestAcc = 0.01
Z = stats.norm.ppf(1-ZTestAcc/2)
print("Running with z test acc of ", 1-ZTestAcc, " with Z star = ", Z)


# consIdx = sys.argv[1]
# conss =  np.linspace(0.3, 0.5, 10)
# cons = conss[int(consIdx)]

# rootDir = 'ConsIdx'+consIdx
rootDir = "pyEmc_test"

# logging.basicConfig(filename=os.path.join(rootDir, 'CLEASE-heatCap.log'), level=logging.INFO)
# np.random.seed(42)  # Set a seed for consistent tests


cmds = ["pair_style eim",
        "pair_coeff * * Na Li /Users/Michael_wang/Documents/venkat/cleaseASEcalc/ffield.eim Na Li"]
rootDir = "/Users/Michael_wang/Documents/venkat/cleaseASEcalc/LiNa"
ASElammps0 = LAMMPSlib(lmpcmds=cmds, keep_alive=True, log_file=os.path.join(rootDir, "LAMMPSloggs0.log"))
ASElammps1 = LAMMPSlib(lmpcmds=cmds, keep_alive=True, log_file=os.path.join(rootDir, "LAMMPSloggs1.log"))
# lmp = lammps()

# print(ASElammps, lmp)


conc = Concentration(basis_elements=[['Li', 'Mg']])
db_name = "LiMg/LiMg-Sep8-first-Batch-hcp.db"
HCPsettings = CEBulk(crystalstructure='hcp',
                  a=3.17,
                  c=5.14,
                  supercell_factor=64,
                  concentration=conc,
                  db_name=db_name,
                  max_cluster_dia=[5.5,5.5,5.5],
                  basis_func_type="polynomial")
with open(db_name + "-eci.json") as f:
    HCPeci = json.load(f)

# conc = Concentration(basis_elements=[['Li', 'Mg']])
db_name = "LiMg/LiMg-Sep8-first-Batch-bcc.db"
BCCsettings = CEBulk(crystalstructure='bcc',
                  a=4.33,
                  supercell_factor=64,
                  concentration=conc,
                  db_name=db_name,
                  max_cluster_dia=[7,7,7],
                  basis_func_type="polynomial")
with open(db_name + "-eci.json") as f:
    BCCeci = json.load(f)

# db_name = "LiNa/LiNa-Sep5-first-Batch-EOS-7-cutoff.db"
# conc = Concentration(basis_elements=[['Li', 'Na']])

# MCsettings = CEBulk(crystalstructure='bcc',
#                   a=3.9,
#                   supercell_factor=64,
#                   concentration=conc,
#                   db_name=db_name,
#                   max_cluster_dia=[7,7,7],
#                   basis_func_type="polynomial")
# with open(db_name + "-eci.json") as f:
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

# print(eci2chem_pot(bf_ls, eci))

# exit()

def get_conc(atoms):
    return np.sum(atoms.numbers == 3) / len(atoms.numbers)

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

        self.bestVar = np.inf
        self.finalPOI = np.inf
        self.bestPOI = np.inf
        self.finalOther = np.inf
        self.bestOther = np.inf

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
        if self.converged:
            return self.finalPOI
        else:
            print("early stop triggered, Non confident best value returned")
            return self.bestPOI

    def get_other(self):
        if self.converged:
            return self.finalOther
        else:
            print("early stop triggered, Non confident best value returned")
            return self.bestOther

    def new_data(self, POI, other):
        POI = POI + (np.random.rand() - .5)*1e-6
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
                        if var < self.bestVar:
                            self.bestVar = var
                            self.bestPOI = (self.POIBins[self.curr_bin] - self.POIBins[bin1]) / n
                            self.bestOther = (self.otherBins[self.curr_bin] - self.otherBins[bin1]) / n
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

                self.corrBins[np.arange(self.nb_bins / 2, dtype=int) + 1] = (self.corrBins[
                                                                                 np.arange(self.nb_bins / 2,
                                                                                           dtype=int) * 2 + 2] +
                                                                             self.corrBins[
                                                                                 np.arange(self.nb_bins / 2,
                                                                                           dtype=int) * 2 + 1]) / 2
                self.corrBins[int(self.nb_bins / 2) + 1:] = 0

                self.curr_bin = int(self.nb_bins / 2 + 1)
                self.granularity = int(self.granularity * 2)
                self.POITmpRecord = np.zeros(self.granularity)

        return self.converged

SCALE = 0.1
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
    # if mu > 3.8:
    DEBUG = 1
    # else:
    #     DEBUG = 1
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
        if BICCounter > 3: break
        if bestP > 5: break

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
        plt.figure()
        plt.errorbar(x, ys, ls='dotted', yerr=np.full(len(ys), SCALE*2))
        plt.plot(x, fn(x))
        plt.plot(np.linspace(x[0], x[-1]+2, 100), fn(np.linspace(x[0], x[-1]+2, 100)), "-o")
        plt.errorbar(x[-1]+np.arange(numPred)+1, fn(x[-1]+np.arange(numPred)+1), yerr=np.sqrt(littlev.diagonal())*2, capsize=5)
        plt.errorbar(x[-1]+1, newY, yerr=np.sqrt(sigma2 + np.sqrt(small_value)), capsize=5)
        plt.show()

    if DEBUG:
        print("sigma: ", np.sqrt(sigma2))
        print("at mu=", mu, "    z score: ", abs(fn(x[-1]+np.arange(numPred)+1) - newY)/np.sqrt(littlev + sigma2 + np.sqrt(small_value)))
    
    DEBUG = 1
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


"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!GS0 MUST HAVE SMALLER MU THAN GS1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
gs0_db_name = "LiMg/LiMg-Mg-gs.db"
db = connect(gs0_db_name)
gs0 = None
for row in db.select(""):
    gs0 = row.toatoms()

# gs0.calc = ASElammps0
gs0 = attach_calculator(HCPsettings, atoms=gs0, eci=HCPeci)
# gs05.calc = EAM(potential='AuCu_Zhou04.eam.alloy')

view(gs0)
print("gs atom E:", gs0.get_potential_energy())

gs1_db_name = "LiMg/LiMg-Li-gs.db"
db = connect(gs1_db_name)
gs1 = None
for row in db.select(""):
    gs1 = row.toatoms()

# gs1.calc = ASElammps1
gs1 = attach_calculator(BCCsettings, atoms=gs1, eci=BCCeci)
view(gs1)

# old_spins = np.copy(gs0.numbers)

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

# TInit, TFinal = 3.2e5, 4e5
# dT = 5e3 * 1e8
# Ts = np.arange(TInit, TFinal+1e-5, dT)

TInit = 200
muInit = -0.2725
dMuMax = 0.004
muRange = [-.4, .1]
mus =  np.arange(muRange[0], muRange[1]+dMuMax*1e-5, dMuMax)
dT = 50# * 1e8
oldDT = np.inf
oldDMu = np.inf

gss = [gs0, gs1]
EPhase = np.array([np.inf,np.inf])
xPhase = np.array([np.inf,np.inf])
mcStarted = np.array([0,0])
lroPhase = [[],[]]
discontPhase = np.array([np.inf,np.inf])
orderArrayPhase = np.array([np.inf,np.inf])

temp = TInit
mu = muInit

phbPrec = 0.003

while 1:
    tmpGs = [gss[0].copy(), gss[1].copy()]
    tmpPOI = np.array([np.inf,np.inf])
    if temp >= 195000:
        print("ha!")
    secondpass = 0
    while 1:
            
        
        # if not discontPhase[0] and not discontPhase[1]:
        #     break
        # if discontPhase[0] == 1 and discontPhase[1] == 1:
        #     break

        discontPhase = [0,0]
        print("Looking for phase transition...")
        # print("mu\tx")
        # if secondpass:
        #     dMuMax /= 2
        # secondpass = 1
        # muBound = np.array([np.inf,np.inf])
        # newMu = mu
        # phase = int(discontPhase[1])
        # # need to increase mu if phase1 discont, decrease mu if phase0 discont
        # dMu = dMuMax * (phase-0.5)*2

        phase = 0
        XMg = []
        deltSpinDetecter = []
        for muMg in mus:
            old_spins = np.copy(gss[phase].numbers)
            systemSize = len(gss[phase].numbers)
            mcSGC = SGCMonteCarlo(gss[phase], temp, symbols=['Li', 'Mg'])
            numOfAu = []
            MaxIterNum = 1000000
            eq = equilibrium_detector(granularity=5, nb_bins=16, prec=phbPrec, patience=6)
            for i in range(MaxIterNum):
                if not i % 100000:
                    print(i)
                mcSGC.run(steps=100, chem_pot={'c1_0': muMg})
                currX = get_conc(gss[phase])
                numOfAu.append(currX)
                # if converged, save qautities of interest, get ready to integrate
                if eq.new_data(POI=currX, other=0) or i == MaxIterNum-1:
                    print("converged after: ", i)
                    # save converged state of [i_tmp, 0] for [i_tmp+1, 0]
                    if DEBUG > 1:
                        plot_stuff(range(i+1), [numOfAu, eq.get_his()], myTitle=[str(temp)])
                        plt.show()
                        view(gss[phase])
                    if DEBUG:
                        print(gss[phase])
                        print("x", eq.get_POI())
                    # saving qautities of interest, if is to be able to collapse code
                    if 1:
                        # lroPhase[phase][-1] = np.sum(old_spins != gss[phase].numbers)
                        deltSpinDetecter.append(np.sum(old_spins != gss[phase].numbers))
                        XMg.append(eq.get_POI())

                    # reset chemical potential
                    tmpNum = gss[phase].numbers
                    mcSGC.run(steps=0, chem_pot={'c1_0': 0})
                    for (a, b) in zip(tmpNum, gss[phase].numbers):
                        assert a == b
                    # exit()
                    break
        
        deltSpinDetecter = np.flip(deltSpinDetecter)
        # print(deltSpinDetecter)
        critialMgIdx = np.inf
        for critialMgIdx, ds in enumerate(deltSpinDetecter):
            # if ds < 1 - 1e-3:
            #     break
            if disconti_detector(ys=deltSpinDetecter[:critialMgIdx], newY=deltSpinDetecter[critialMgIdx], maxYLength=20) > Z:
                break
        critialMgIdx = len(mus) - critialMgIdx -1
        print(critialMgIdx)

        # plt.plot(mus, XMg)
        # plt.vlines(mus[critialMgIdx], 0, 1)
        # plt.show()

        phase = 1
        XLi = []
        deltSpinDetecter = []
        for muLi in np.flip(mus):
            old_spins = np.copy(gss[phase].numbers)
            systemSize = len(gss[phase].numbers)
            mcSGC = SGCMonteCarlo(gss[phase], temp, symbols=['Li', 'Mg'])
            numOfAu = []
            MaxIterNum = 1000000
            eq = equilibrium_detector(granularity=5, nb_bins=16, prec=phbPrec, patience=6)
            for i in range(MaxIterNum):
                if not i % 100000:
                    print(i)
                mcSGC.run(steps=100, chem_pot={'c1_0': muLi})
                currX = get_conc(gss[phase])
                numOfAu.append(currX)
                # if converged, save qautities of interest, get ready to integrate
                if eq.new_data(POI=currX, other=0) or i == MaxIterNum-1:
                    print("converged after: ", i)
                    # save converged state of [i_tmp, 0] for [i_tmp+1, 0]
                    if DEBUG > 1:
                        plot_stuff(range(i+1), [numOfAu, eq.get_his()], myTitle=[str(temp)])
                        plt.show()
                        view(gss[phase])
                    if DEBUG:
                        print(gss[phase])
                        print("x", eq.get_POI())
                    # saving qautities of interest, if is to be able to collapse code
                    if 1:
                        # lroPhase[phase][-1] = np.sum(old_spins != gss[phase].numbers)
                        deltSpinDetecter.append(np.sum(old_spins != gss[phase].numbers))
                        XLi.append(eq.get_POI())
                        old_spins = np.copy(gss[phase].numbers)

                    # reset chemical potential
                    tmpNum = gss[phase].numbers
                    mcSGC.run(steps=0, chem_pot={'c1_0': 0})
                    for (a, b) in zip(tmpNum, gss[phase].numbers):
                        assert a == b
                    # exit()
                    break
        
        deltSpinDetecter = np.flip(deltSpinDetecter)
        # print(deltSpinDetecter)
        critialLiIdx = np.inf
        for critialLiIdx, ds in enumerate(deltSpinDetecter):
            # if ds > 1e-3:
            #     break
            if disconti_detector(ys=deltSpinDetecter[:critialLiIdx], newY=deltSpinDetecter[critialLiIdx], maxYLength=20) > Z:
                break
        # print(critialLiIdx)

        # plt.plot(np.flip(mus), XLi)
        # plt.vlines(mus[critialLiIdx], 0, 1)
        # plt.show()

        critialIdx = (critialMgIdx + critialLiIdx)/2
        if critialIdx % 2 == 0:
            tmpPOI[0] = XMg[int(critialIdx)]
            tmpPOI[1] = np.flip(XLi)[int(critialIdx)]
            citMu = mus[int(critialIdx)]
        else:
            tmpPOI[0] = (XMg[int(critialIdx)] + XMg[int(critialIdx)+1])/2
            tmpPOI[1] = (np.flip(XLi)[int(critialIdx)] + np.flip(XLi)[int(critialIdx)+1])/2
            citMu = (mus[int(critialIdx)] + mus[int(critialIdx)+1])/2

        plt.plot(mus, XMg)
        plt.vlines(mus[critialMgIdx], 0, 1)
        plt.plot(np.flip(mus), XLi)
        plt.vlines(mus[critialLiIdx], 0, 1)
        plt.vlines(citMu, 0, 1)
        plt.hlines(tmpPOI[0], muRange[0], muRange[1])
        plt.hlines(tmpPOI[1], muRange[0], muRange[1])
        plt.show()

        break
        # muBound[1] = newMu
        # mu = (muBound[0] + muBound[1]) / 2
        # oldDT = np.inf
        # oldDMu = np.inf
        # gss[0] = attach_calculator(MCsettings, atoms=tmpGs[0].copy(), eci=eci)
        # gss[1] = attach_calculator(MCsettings, atoms=tmpGs[1].copy(), eci=eci)
        # # gss[0] = tmpGs[0].copy()
        # # gss[0].calc = ASElammps0
        # # gss[1] = tmpGs[1].copy()
        # # gss[1].calc = ASElammps1

        # for phase in range(2):
        #     systemSize = len(gss[phase].numbers)
        #     mcSGC = SGCMonteCarlo(gss[phase], temp, symbols=['Li', 'Na'], observe_singlets=False)
        #     E = []
        #     numOfAu = []
        #     MaxIterNum = 1000000
        #     avgRecorder = []
        #     eq = equilibrium_detector(granularity=5, nb_bins=16, prec=phbPrec, patience=6)
        #     for i in range(MaxIterNum):
        #         if not i % 100000:
        #             print(i)
        #         mcSGC.run(steps=100, chem_pot={'c1_0': mu})
        #         currE = mcSGC.get_thermodynamic_quantities()['sgc_energy']
        #         # currSCGE = mcSGC.get_thermodynamic_quantities()['singlet_energy']
        #         # print(gs05.get_potential_energy(), currE, currSCGE)
        #         E.append(currE)
        #         currX = get_conc(gss[phase])
        #         numOfAu.append(currX)

        #         if not i % avgWindowWidth:
        #             avgRecorder.append(np.average(E[-100:]))

        #         # if converged, save qautities of interest, get ready to integrate
        #         if eq.new_data(POI=currX, other=currE) or i == MaxIterNum-1:
        #         # if i > 2000 or i == MaxIterNum - 1:
        #             print("converged after: ", i, "at T =", temp)
        #             # save converged state of [i_tmp, 0] for [i_tmp+1, 0]
        #             if DEBUG > 1:
        #                 plot_stuff(range(i+1), [E, numOfAu, eq.get_his()], myTitle=[str(temp)])
        #                 plt.show()
        #                 view(gss[phase])
        #             if DEBUG:
        #                 print(gss[phase])
        #                 print("E", eq.get_other(), "x", eq.get_POI())
        #             # saving qautities of interest, if is to be able to collapse code
        #             if 1:
        #                 EPhase[phase] = eq.get_other()/systemSize
        #                 xPhase[phase] = eq.get_POI()
        #                 tmpPOI[phase] = eq.get_POI()
        #                 # if secondpass:
        #                 #     # lroPhase[phase][-1] = np.sum(old_spins != gss[phase].numbers)
        #                 #     lroPhase[phase][-1] = eq.get_POI()

        #                 # else:
        #                 #     # lroPhase[phase].append(np.sum(old_spins != gss[phase].numbers))
        #                 #     lroPhase[phase].append(eq.get_POI())
        #                 old_spins = np.copy(gss[phase].numbers)

        #             # reset chemical potential
        #             tmpNum = gss[phase].numbers
        #             mcSGC.run(steps=0, chem_pot={'c1_0': 0})
        #             for (a, b) in zip(tmpNum, gss[phase].numbers):
        #                 assert a == b
        #             # exit()
        #             break
        #     if len(lroPhase[phase]) * len(lroPhase[1-phase]):
        #         discontPhase[phase] = (disconti_detector(ys=lroPhase[phase],newY=tmpPOI[phase], mu = mu) > disconti_detector(ys=lroPhase[1-phase],newY=tmpPOI[phase], mu = mu))
        #     else:
        #         discontPhase[phase] = 0
    
    lroPhase[0].append(tmpPOI[0])
    lroPhase[1].append(tmpPOI[1])

    logF = open("phbLog-limg-CLEASE-acc0.002-XXL.txt", 'a')
    logF.writelines(str(tmpPOI[0]) + '\t' + str(tmpPOI[1]) + '\n')
    logF.close()
    
    if discontPhase[0] and discontPhase[1]:
        print("third phase appeared!")
        break
    if abs(xPhase[0] - xPhase[1]) < 0.01:
        print("phase boundray merged!")
        break
    if temp >= 1000:
        break
    temp += dT
    # dMu = kB*temp*(EPhase[1]-EPhase[0])/(xPhase[1]-xPhase[0]) - (mu-muInit)*kB*temp
    # dMu *= (1/kB/temp-1/kB/(temp-dT)) 
    
    # if oldDT == np.inf:
    #     temp += dT
    #     # mu += dMuMax
    #     mu += dMu
    # else:
    #     temp += 1.5 * dT - 0.5 * oldDT
    #     mu += 1.5 * dMu - 0.5 * oldDMu
    # oldDT = dT
    # oldDMu = dMu
    # dMuMax = abs(dMu)

    








import sys
sys.path.append("/nfs/turbo/coe-venkvis/ziqiw-turbo/PD/solid/clease")

import json
import logging
import sys, os, time
import yaml
import math
import random

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
from ase.units import kB

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

def get_conc(atoms):
    return np.sum(atoms.numbers == 3) / len(atoms)

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


# kB = 8.617333262e-5 * 0.0433634 * 1e10
np.set_printoptions(precision=3, suppress=True)

with open('LiMg-example.yaml', 'r') as file:
    options = yaml.safe_load(file)

DEBUG = options["EMC"]["DEBUG"]

if "LAMMPS" in options:
        from ase.calculators.lammpslib import LAMMPSlib
        print("LAMMPS calc!")
        ASElammps = LAMMPSlib(**options["LAMMPS"])

GroundStates = options["GroundStates"]
gs_db_names = options["EMC"]["gs_db_names"]
muInit, muFinal = options["EMC"]["muInit"], options["EMC"]["muFinal"]
dMu = options["EMC"]["dMu"]

dMu = abs(dMu)
if (muFinal - muInit)*dMu < 0:
    dMu = -dMu

gsE = [0,0]
for gsIdx, gs_name in enumerate(GroundStates):
    
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

    # gs05.calc = ASElammps
    gs05 = attach_calculator(MCsettings, atoms=gs05, eci=eci)
    
    tmpLi = gs05.copy()
    tmpLi.calc = gs05.calc
    tmpLi.numbers[:] = 3
    gsE[0] = tmpLi.get_potential_energy()/len(tmpLi)
    tmpLi.numbers[:] = 12
    gsE[1] = tmpLi.get_potential_energy()/len(tmpLi)
    # gsE = [0,0]
        
    gs05.info["gsE"] = gsE
    
    TInit, TFinal = options["EMC"]["TInit"], options["EMC"]["TFinal"]
    dT = options["EMC"]["dT"]
    Ts = np.arange(TInit, TFinal+1e-5, dT)
    # Ts = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
    if gsIdx == 0:
        mus = np.arange(muInit, muFinal+dMu*1e-5, dMu)
    elif gsIdx == 1:
        muInit, muFinal = muFinal, muInit
        dMu = -dMu
        mus = np.arange(muInit, muFinal+dMu*1e-5, dMu)
    
    lte = LTE(formation=False)
    lte.set_gs_atom(gs05)
    phi_lte = lte.get_E(T=Ts[0], mu=mus[0])
    
    print("######################################")
    print("at T =", Ts[0], gs_name, "E_lte = ", phi_lte)
    print("######################################")
    
    phiTable = np.zeros((len(Ts), len(mus)))
    phiTable[0,0] = phi_lte 
    print(phiTable.shape)
    ETable = np.zeros((len(Ts), len(mus)))
    XTable = np.zeros((len(Ts), len(mus)))
    deltSpinDetecter = np.zeros((len(Ts), len(mus)))
    systemSize = len(gs05)
    old_spins = np.copy(gs05.numbers)
    
    for iT, currT in enumerate(Ts):
        for iMu, currMu in enumerate(mus):
            eq = equilibrium_detector(granularity=100, nb_bins=16, prec=options["EMC"]["error"], patience=6)
            MaxIterNum = 1000000
            print("T = ", currT, "mu = ", currMu)
            for i in range(MaxIterNum):
                currOldE = gs05.get_potential_energy() - (gsE[0] * get_conc(gs05) + gsE[1] * (1-get_conc(gs05)) + currMu * get_conc(gs05)) * len(gs05)
                flipIdx = int(np.random.uniform(low=0, high=len(gs05)-1))
                if gs05.numbers[flipIdx] == 3:
                    gs05.numbers[flipIdx] = 12
                else:
                    gs05.numbers[flipIdx] = 3
                currNewE = gs05.get_potential_energy() - (gsE[0] * get_conc(gs05) + gsE[1] * (1-get_conc(gs05)) + currMu * get_conc(gs05)) * len(gs05)
                currE = currNewE
                
                if currNewE > currOldE:
                    energy_diff = currNewE - currOldE
                    probability = math.exp(-energy_diff / kB / currT)
                    if random.random() > probability:
                        currE = currOldE
                        if gs05.numbers[flipIdx] == 3:
                            gs05.numbers[flipIdx] = 12
                        else:
                            gs05.numbers[flipIdx] = 3
                
                if eq.new_data(POI=get_conc(gs05), other=currE) or i == MaxIterNum-1:
                    print("converged after: ", i)
                    if iT == 0:
                        prevGs05 = gs05.copy()
                    
                    ETable[iT, iMu] = eq.get_other()/systemSize
                    XTable[iT, iMu] = eq.get_POI()
                    deltSpinDetecter[iT, iMu] = np.sum(old_spins != gs05.numbers)
                    old_spins = np.copy(gs05.numbers)
                    break
            
            if DEBUG:
                if len(mus) == 1:
                    print(deltSpinDetecter[:iT+1, 0])
                else:
                    print(deltSpinDetecter[iT, :iMu+1])
                    
            if iT == 0 and iMu == 0:
                # phiTable[0,0] = 0
                continue
            
            if iMu == 0:
                print("#######################################################################################################")
                prevB, currB = 1/kB/(currT - dT), 1/kB/currT
                phiTable[iT, iMu] = phiTable[iT - 1, iMu] * prevB / currB + (ETable[iT, iMu] + ETable[iT - 1, iMu]) / 2 * (currB - prevB) / currB
                # phiTable[iT, iMu] = phiTable[iT - 1, iMu] * prevB / currB + (ETable[iT, iMu] + ETable[iT - 1, iMu]) / 2 * (currB - prevB) / currB
                print(phiTable[iT, iMu])
                print(phiTable[iT - 1, iMu],  phiTable[iT - 1, iMu] * prevB / currB ,(ETable[iT, iMu] + ETable[iT - 1, iMu]) / 2 * (currB - prevB) / currB)
                print(ETable[iT, iMu] , ETable[iT - 1, iMu])
            else:
                phiTable[iT, iMu] = phiTable[iT, iMu - 1] - (XTable[iT, iMu] + XTable[iT, iMu - 1]) / 2 * dMu
            
        
        gs05 = prevGs05.copy()
        gs05.calc = ASElammps
        gs05.calc.reset()
        gs05.info["gsE"] = gsE
    
    if len(Ts) == 1:                
        plt.figure(1)
        plt.plot(mus, phiTable[-1], '-o')
        plt.title(gs_name)
        # plt.show()
        plt.figure(2)
        plt.plot(mus, XTable[-1], '-o')
        plt.title(gs_name)
    else:
        plt.figure(1)
        plt.plot(Ts, phiTable[:,0], '-o')
        plt.title(gs_name)
        # plt.show()
        plt.figure(2)
        plt.plot(Ts, XTable[:,0], '-o')
        plt.title(gs_name)
    
plt.figure(1)
plt.title("phi vs T")
plt.xlabel("T")
plt.ylabel("phi")
plt.legend(["BCC", "HCP"])
# plt.xlim([-0.25, 0.25])
# plt.ylim([-0.5, 0])
plt.savefig('T_phi.png')

plt.figure(2)
plt.title("Li concentration vs. mu")
plt.xlabel("mu")
plt.ylabel("Li concentration")
plt.legend(["BCC Li -> BCC Mg", "HCP Mg -> HCP Li"])
# plt.xlim([-0.25, 0.25])
plt.savefig('mu_x.png')
plt.show()

print(ETable)
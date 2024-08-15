import sys
sys.path.append("/nfs/turbo/coe-venkvis/ziqiw-turbo/mint-PD/PD/clease")

import json
import logging
import sys, os, time
import yaml
import math
import random
import time
import argparse

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
from ase.optimize import FIRE
from ase.constraints import UnitCellFilter

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

def generateBatchInput(input, Ts):
    slurm_script = f"""#!/bin/bash
#SBATCH --array=0-{len(Ts)-1}   # Array index range
#SBATCH -J emc_%a # Job name
#SBATCH -n 1 # Number of total cores
#SBATCH -N 1 # Number of nodes
#SBATCH --time=00-8:00:00
#SBATCH -p venkvis-cpu
#SBATCH --mem=2000 # Memory pool for all cores in MB
#SBATCH -e outNerr/emc_%A_%a.err #change the name of the err file 
#SBATCH -o outNerr/emc_%A_%a.out # File to which STDOUT will be written %j is the job #

source /nfs/turbo/coe-venkvis/ziqiw-turbo/.bashrc
conda activate /nfs/turbo/coe-venkvis/ziqiw-turbo/conda/casm
cd /nfs/turbo/coe-venkvis/ziqiw-turbo/mint-PD/PD

# Define the array of custom values
VALUES=({" ".join(map(str, Ts))})

# Get the value corresponding to the current task ID
MY_VALUE=${{VALUES[$SLURM_ARRAY_TASK_ID]}}

echo "Running task $SLURM_ARRAY_TASK_ID with value $MY_VALUE"

# Example: Run a Python script with the selected value
python fmc.py --input="{input}" --batch=true --init=false --temp=$MY_VALUE
"""

    with open(f"runBatch.sh", 'w') as f:
        f.write(slurm_script)
    

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



parser = argparse.ArgumentParser(description="Process some command-line arguments.")

# Add the arguments
parser.add_argument('-i', '--input', type=str, default='LiNa-example.yaml', help='input yaml filename')
parser.add_argument('-b', '--batch', type=str, default='False', help='Set a (default: True)')
parser.add_argument('--init', type=str, default='True', help='Set b (default: True)')
parser.add_argument('-t', '--temp', type=str, default='', help='temperature needed if init=false')

# Parse the arguments
args = parser.parse_args()

# Access the values
inputFileName = args.input

if args.batch.lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']:
    batchMode = True
    if args.init.lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']:
        batchInit = True
        os.makedirs(f"{inputFileName[:-5]}-reuslt", exist_ok=True)
        batchTemp = ''
    elif args.init.lower() in ['false', '0', 'f', 'n', 'no', 'nope', 'nah', 'never']:
        batchInit = False
        try:
            _ = float(args.temp)
            batchTemp = args.temp
        except:
            raise ValueError(f"Invalid value for temp: {args.temp}")
    else:
        raise ValueError(f"Invalid value for batch init: {args.init}")
elif args.batch.lower() in ['false', '0', 'f', 'n', 'no', 'nope', 'nah', 'never']:
    batchMode = False
    batchInit = True
    batchTemp = ''
else:
    raise ValueError(f"Invalid value for batch mode: {args.batch}")


# kB = 8.617333262e-5 * 0.0433634 * 1e10
np.set_printoptions(precision=7, suppress=True)

with open(inputFileName, 'r') as file:
    options = yaml.safe_load(file)
    
print(f"input loaded from {inputFileName}, batchMode = {batchMode}, batchInit = {batchInit}, batchTemp = {batchTemp}")

DEBUG = options["EMC"]["DEBUG"]

if "LAMMPS" in options:
        from ase.calculators.lammpslib import LAMMPSlib
        print("LAMMPS calc!")
        ASElammps = LAMMPSlib(**options["LAMMPS"])

GroundStates = options["GroundStates"]
gs_db_names = options["EMC"]["gs_db_names"]
muInit, muFinal = options["EMC"]["muInit"], options["EMC"]["muFinal"]

if batchMode and batchInit:
    dMu = 1e10
else:
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
    
    # gs05.set_cell(gs05.cell*1.1, scale_atoms=True)
    
    # gs05 = attach_calculator(MCsettings, atoms=gs05, eci=eci)
    
    gs05.calc = ASElammps
    ucf = UnitCellFilter(gs05)
    opt = FIRE(ucf)
    opt.run(fmax=0.02)
    
    gsE[gsIdx] = gs05.get_potential_energy()/len(gs05)
    
startt = time.time()

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

    # gs05 = attach_calculator(MCsettings, atoms=gs05, eci=eci)
    gs05.calc = ASElammps

    # tmpLi = gs05.copy()
    # tmpLi.calc = gs05.calc
    # tmpLi.numbers[:] = 3
    # gsE[0] = tmpLi.get_potential_energy()/len(tmpLi)
    # tmpLi.numbers[:] = 12
    # gsE[1] = tmpLi.get_potential_energy()/len(tmpLi)
    # gsE = [0,0]
        
    gs05.info["gsE"] = gsE

    if gsIdx == 0:
        mus = np.arange(muInit, muFinal+dMu*1e-5, dMu)
    elif gsIdx == 1:
        muInit, muFinal = muFinal, muInit
        dMu = -dMu
        mus = np.arange(muInit, muFinal+dMu*1e-5, dMu)
    
    if batchMode:
        if batchInit:
            TInit, TFinal = options["EMC"]["TInit"], options["EMC"]["TFinal"]
            dT = options["EMC"]["dT"]
            Ts = np.arange(TInit, TFinal+1e-5, dT)
            
            generateBatchInput(Ts)
            
            lte = LTE(formation=False)
            lte.set_gs_atom(gs05)
            phi_lte = lte.get_E(T=Ts[0], mu=mus[0])
        else:
            with open(f"{inputFileName[:-5]}-reuslt/{gs_name}-startingPoints.json", 'r') as f:
                TsDict = json.load(f)
            Ts = [float(batchTemp)]
            
            # this lte is for DEBUG ONLY!!!
            try:
                lte = LTE(formation=False)
                lte.set_gs_atom(gs05)
                phi_lte_lte = lte.get_E(T=Ts[0], mu=mus[0])
            except:
                phi_lte_lte = 0
                print("debugging LTE fail to run")
            
            phi_lte = TsDict[batchTemp]
            print("REFREFREFREFREFREFREFREFREFREFREF")
            print("at T =", Ts[0], gs_name, "loaded_lte = ", phi_lte, "calulated_lte = ", phi_lte_lte)
            print("REFREFREFREFREFREFREFREFREFREFREF")
    else:
        TInit, TFinal = options["EMC"]["TInit"], options["EMC"]["TFinal"]
        dT = options["EMC"]["dT"]
        Ts = np.arange(TInit, TFinal+1e-5, dT)
            
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
            eq = equilibrium_detector(granularity=200, nb_bins=16, prec=options["EMC"]["error"], patience=6)
            MaxIterNum = 1000000
            print("T = ", currT, "mu = ", currMu)
            for i in range(MaxIterNum):
                
                # in order to calculate the acceptance rate
                acceptanceRecord = np.zeros((1000))
                VScaleRange = 0.1
                
                if i % 100 == 0:
                    print(i)
                
                # calculate the energy for the old strucutre
                ucf = UnitCellFilter(gs05)
                opt = FIRE(ucf, logfile=None)
                converged = opt.run(fmax=0.02, steps=500)
                if not converged:
                    print(f"FIRE MAXSTEP REACHED at {i} for oldE!!!")
                currOldE = gs05.get_potential_energy() - (gsE[0] * get_conc(gs05) + gsE[1] * (1-get_conc(gs05)) + currMu * get_conc(gs05)) * len(gs05)
                oldSysXyz = gs05.positions.copy()
                # randomly flip one
                flipIdx = int(np.random.uniform(low=0, high=len(gs05)-1))
                if gs05.numbers[flipIdx] == 3:
                    gs05.numbers[flipIdx] = 12
                else:
                    gs05.numbers[flipIdx] = 3
                # calculate the new energy
                converged = opt.run(fmax=0.02, steps=500)
                if not converged:
                    print(f"FIRE MAXSTEP REACHED at {i} for newE!!!")
                currNewE = gs05.get_potential_energy() - (gsE[0] * get_conc(gs05) + gsE[1] * (1-get_conc(gs05)) + currMu * get_conc(gs05)) * len(gs05)
                # temporarily accept the new energy
                currE = currNewE
                
                
                if currNewE > currOldE:
                    energy_diff = currNewE - currOldE
                    probability = math.exp(-energy_diff / kB / currT)
                    if random.random() > probability:
                        # DAMN!!!!! revert
                        currE = currOldE
                        gs05.positions = oldSysXyz.copy()
                        if gs05.numbers[flipIdx] == 3:
                            gs05.numbers[flipIdx] = 12
                        else:
                            gs05.numbers[flipIdx] = 3
                
                # # currE is now the accepted energy and gs05 is now the accepted strucutre
                # # randomly pick a scaling factor
                # scaleScale = np.random.uniform(low=1-VScaleRange, high=1+VScaleRange)
                # # scale the system
                # gs05.set_cell(gs05.cell*scaleScale, scale_atoms=True)
                # # calculate new energy
                # currNewE = gs05.get_potential_energy() - (gsE[0] * get_conc(gs05) + gsE[1] * (1-get_conc(gs05)) + currMu * get_conc(gs05)) * len(gs05)
                # # record the old E
                # currOldE = currE
                # # temporarily accept the new E
                # currE = currNewE
                # acceptanceRecord[i%1000] = 1
                # if currNewE > currOldE:
                #     energy_diff = currNewE - currOldE
                #     probability = math.exp(-energy_diff / kB / currT)
                #     if random.random() > probability:
                #         # DAMN!!!!! revert
                #         currE = currOldE
                #         acceptanceRecord[i%1000] = 0
                #         gs05.set_cell(gs05.cell/scaleScale, scale_atoms=True)
                
                # if i % 1000 == 0:
                #     acptRate = np.mean(acceptanceRecord)
                #     print(acptRate)
                #     if acptRate > 0.75:
                #         VScaleRange*1.1
                #     if acptRate < 0.25:
                #         if VScaleRange/1.1 > 1:
                #             VScaleRange/1.1
                            
                if eq.new_data(POI=get_conc(gs05), other=currE) or i == MaxIterNum-1:
                    print("converged after: ", i)
                    if iT == 0:
                        prevGs05 = gs05.copy()
                    
                    ETable[iT, iMu] = eq.get_other()/systemSize
                    XTable[iT, iMu] = eq.get_POI()
                    deltSpinDetecter[iT, iMu] = np.sum(old_spins != gs05.numbers)
                    old_spins = np.copy(gs05.numbers)
                    print(gs05.cell)
                    print(gs05.get_volume())
                    break
            if DEBUG:
                if len(mus) == 1:
                    print(deltSpinDetecter[:iT+1, 0])
                    if batchMode:
                        f = open(f"{inputFileName[:-5]}-reuslt/fmc-mu-{mus[0]}.out", "a")
                    else:
                        f = open("fmc-mu.out", "a")
                    for xxx in deltSpinDetecter[:iT+1, 0]:
                        f.write(f"{xxx} ")
                    f.write("\n")
                    f.close()
                else:
                    print(deltSpinDetecter[iT, :iMu+1])
                    if batchMode:
                        f = open(f"{inputFileName[:-5]}-reuslt/fmc-T-{Ts[0]}.out", "a")
                    else:
                        f = open("fmc-T.out", "a")
                    for xxx in deltSpinDetecter[iT, :iMu+1]:
                        f.write(f"{xxx} ")
                    f.write("\n")
                    f.close()
                    
            if iT == 0 and iMu == 0:
                # phiTable[0,0] = 0
                continue
            
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
        if batchMode:
            np.save(f"{inputFileName[:-5]}-reuslt/{gs_name}-{Ts[0]}-phiTable.npy", phiTable[-1])
        plt.title(gs_name)
        # plt.show()
        plt.figure(2)
        plt.plot(mus, XTable[-1], '-o')
        plt.title(gs_name)
    else:
        startingPoints = {}
        if batchMode:
            for idx, ts in enumerate(Ts):
                startingPoints[str(ts)] = phiTable[idx,0]
            with open(f"{inputFileName[:-5]}-reuslt/{gs_name}-startingPoints.json", 'w') as f:
                json.dump(startingPoints, f)
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
if batchMode:
    plt.savefig(f"{inputFileName[:-5]}-reuslt/T_phi_{batchTemp}.png")
else:
    plt.savefig("T_phi.png")

plt.figure(2)
plt.title("Li concentration vs. mu")
plt.xlabel("mu")
plt.ylabel("Li concentration")
plt.legend(["BCC Li -> BCC Mg", "HCP Mg -> HCP Li"])
# plt.xlim([-0.25, 0.25])
if batchMode:
    plt.savefig(f"{inputFileName[:-5]}-reuslt/mu_x_{batchTemp}.png")
else:
    plt.savefig("mu_x.png")
plt.show()

print(ETable)

print(f"walltime: {time.time() - startt}")
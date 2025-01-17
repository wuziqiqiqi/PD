import numpy as np
from clease.calculator import attach_calculator
from ase.visualize import view
from ase.optimize import FIRE
from ase.constraints import UnitCellFilter
from clease.calculator import Clease

class LTE:
    def __init__(self, myCalc = None, formation = False):
        self.gs_atom = None
        self.gs_E = None
        self.N = None
        self.x = None
        self.gsE0 = 0
        self.gsE1 = 0
        # -1.6483617991, -1.5493150298
        self.myCalc = myCalc
        self.formation = formation

    def set_gs_atom(self, atom):
        """
        This function set ground state structure, system N, x, and calculate the gs energy
        """
        self.gs_atom = atom
        self.N = len(atom)
        self.x = np.sum(self.gs_atom.numbers == 3)/self.N
        self.gsE0 = atom.info["gsE"][0]
        self.gsE1 = atom.info["gsE"][1]
        NLi = np.sum(self.gs_atom.numbers == 3)
        NMg = len(self.gs_atom) - NLi
        
        # if NMg == 0:
        #     self.gs_atom.numbers[:] = 12
        # else:
        #     self.gs_atom.numbers[:] = 3
        if not isinstance(self.gs_atom.calc, Clease):
            ucf = UnitCellFilter(atom)
            opt = FIRE(ucf)
            opt.run(fmax=0.02)        
        if self.formation:
            self.gs_E = atom.get_potential_energy()/len(atom)
        else:
            self.gs_E = (atom.get_potential_energy() - NLi * self.gsE0 - NMg * self.gsE1)/len(atom)
        print("gs_E =", self.gs_E)
        # tmp = atom.copy()
        # tmp.calc = self.myCalc
        # ucf = UnitCellFilter(tmp)
        # opt = FIRE(ucf, logfile=None)
        # opt.run(fmax=0.02)
        # print(atom.get_potential_energy()/self.N, "################")
        # self.gs_E = tmp.get_potential_energy()/self.N - (self.x+1)/2 * self.Li - (1-(self.x+1)/2) * self.Mg
        pass


    def get_E(self, T, mu):
        # view(self.gs_atom)
        phi = self.gs_E - mu * self.x
        print("phi0 = ", phi)
        kb = 8.617333262e-5
        beta = 1/kb/T
        for i in range(self.N):
            # alter one spin
            currSpecies = self.gs_atom.numbers[i]
            if currSpecies == 3: 
                self.gs_atom.numbers[i] = 12
                dEta = -1
            if currSpecies == 12: 
                self.gs_atom.numbers[i] = 3
                dEta = 1
            
            # need new copy!!!!!!!!!!!!!!!!
            tmp = self.gs_atom.copy()
            tmp.calc = self.gs_atom.calc
            if not isinstance(tmp.calc, Clease):
                ucf = UnitCellFilter(tmp)
                opt = FIRE(ucf)
                opt.run(fmax=0.02)
            
            NLi = np.sum(self.gs_atom.numbers == 3)
            NMg = len(self.gs_atom.numbers) - NLi
            if self.formation:
                newE = self.gs_atom.get_potential_energy()
            else:
                newE = self.gs_atom.get_potential_energy() - NLi * self.gsE0 - NMg * self.gsE1
            # tmp = self.gs_atom.copy()
            # tmp.calc = self.myCalc
            # ucf = UnitCellFilter(tmp)
            # opt = FIRE(ucf, logfile=None)
            # opt.run(fmax=0.02)
            # newE = tmp.get_potential_energy() - ((self.x+1)/2 * self.Li + (1-(self.x+1)/2) * self.Mg) * self.N
            dE = newE - self.gs_E * self.N  # or the other way around? NO! LOL
            # dEta = np.sum(self.gs_atom.numbers == 11) - self.x * self.N
            # if mu > 0:
            #     dEta = -1
            # else:
            #     dEta = 1
            # dEta = 1

            tmp = (mu * dEta - dE) * beta

            print(f"mu = {mu}, dEta = {dEta}, dE = {dE}, tmp = {tmp}")
            assert tmp < 0, "structure at mu = " + str(mu) + "is not minimum"

            phi -= np.exp(tmp)/beta/self.N
            
            print(f"newE = {newE}, dE = {dE}, tmp = {tmp}, phi = {phi}, dphi = {np.exp(tmp)/beta/self.N}")

            # reset
            self.gs_atom.numbers[i] = currSpecies

        return phi
    
class MF:
    def __init__(self):
        self.gs_atom = None
        self.gs_E = None
        self.N = None
        self.x = None

    def set_gs_atom(self, atom, IsingCalc):
        self.gs_atom = atom
        self.N = len(atom.numbers)
        self.x = np.sum(self.gs_atom.numbers == 79)/self.N * 2 - 1

        self.q = 3
        self.IsingCalc = IsingCalc
        self.J = IsingCalc.J
        self.h = IsingCalc.h
        # self.hEff = self.h + 2 * self.q * self.J * self.x
    
    def get_F(self, T, x):
        x = x * 2 - 1
        kb = 8.617333262e-5
        beta = 1/kb/T
        hEff = self.h + 2 * self.q * self.J * x
        a = self.N * self.q * self.J * x**2
        b = self.N * np.log(2) / beta
        if abs(beta * hEff) > 10:
            c = self.N * (abs(beta * hEff)-0.69314718) / beta
        else:
            c = self.N * np.log(np.cosh(beta * hEff)) / beta
        F = a - b - c
        return F
    
    def get_phi(self, T, x, mu):
        F = self.get_F(T, x) / self.N
        print("F =", F)
        phi = F - mu * x
        print("phi =", phi)
        return phi


    


import numpy as np
from clease.calculator import attach_calculator
from ase.visualize import view
from ase.optimize import BFGS
from ase.constraints import UnitCellFilter

class LTE:
    def __init__(self, myCalc = None):
        self.gs_atom = None
        self.gs_E = None
        self.N = None
        self.x = None
        self.Li = -1.6483617991
        self.Mg = -1.5493224718099385
        # -1.6483617991, -1.5493150298
        self.myCalc = myCalc

    def set_gs_atom(self, atom):
        """
        This function set ground state structure, system N, x, and calculate the gs energy
        """
        self.gs_atom = atom
        self.N = len(atom.numbers)
        self.x = np.sum(self.gs_atom.numbers == 3)/self.N*2-1
        self.gs_E = atom.get_potential_energy()/self.N
        # tmp = atom.copy()
        # tmp.calc = self.myCalc
        # ucf = UnitCellFilter(tmp)
        # opt = BFGS(ucf, logfile=None)
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
            if currSpecies == 3: self.gs_atom.numbers[i] = 12
            if currSpecies == 12: self.gs_atom.numbers[i] = 3

            newE = self.gs_atom.get_potential_energy()
            # tmp = self.gs_atom.copy()
            # tmp.calc = self.myCalc
            # ucf = UnitCellFilter(tmp)
            # opt = BFGS(ucf, logfile=None)
            # opt.run(fmax=0.02)
            # newE = tmp.get_potential_energy() - ((self.x+1)/2 * self.Li + (1-(self.x+1)/2) * self.Mg) * self.N
            dE = newE - self.gs_E * self.N  # or the other way around? NO! LOL
            # dEta = np.sum(self.gs_atom.numbers == 11) - self.x * self.N
            if mu > 0:
                dEta = -1
            else:
                dEta = 1
            # dEta = 1

            tmp = (mu * dEta - dE) * beta

            assert tmp < 0, "structure at mu = " + str(mu) + "is not minimum"

            phi -= np.exp(tmp)/beta/self.N

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


    


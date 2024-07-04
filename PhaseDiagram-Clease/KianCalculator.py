from ase.calculators.calculator import Calculator
from ase.db import connect
import numpy as np
import json
import re


class KianCalc(Calculator):
    """Calculator that calculates the total energy of a system by summing
    the energies of individual atoms.
    """

    implemented_properties = ['energy']

    def __init__(self, energyFile, formationEBasis = None, endPointE = None, byOrder = False):
        """Initialize the calculator.

        Parameters:
        atoms: ASE Atoms object
            The system of atoms.
        atom_energies: list of float
            List of energies for each atom in the system.
        """
        Calculator.__init__(self)
        self.atoms = None
        self.currAtomSymbol = None
        self.atom_energies = None
        self.atom_energy_data = None
        self.atom_force_data = None
        energyEndPoints = [np.inf,np.inf]
        toExclude = []
        self.byOrder = byOrder
        if energyFile[-2:] == "db":
            tmpDb = connect(energyFile)
            tmpDictE = {}
            tmpDictF = {}
            for idx, row in enumerate(tmpDb.select("")):
                if idx+1 not in toExclude:
                    atoms = row.toatoms()
                    # if row.formula == "Li64":
                    #     tmpE = self.energy_correction(row.formula, -196.907*80)
                    #     print(row.formula, tmpE)
                    # else:
                    #     tmpE = self.energy_correction(row.formula, row.energy)
                    #     print(row.formula, tmpE)
                    # tmpDictE[row.formula] = tmpE
                    tmpDictE[atoms.symbols.formula._formula] = row.energy
                    # tmpDictF[atoms.symbols.formula._formula] = row.forces

                    if formationEBasis is not None:
                        if endPointE is not None:
                            energyEndPoints[0] = endPointE[0]
                            energyEndPoints[1] = endPointE[1]
                        else:
                            pattern = r'[0-9]'
                            endPointTester = re.sub(pattern, '', row.formula)
                            if endPointTester == formationEBasis[0]:
                                # aaa = row.energy
                                # bbb = len(row.numbers)
                                # ccc = row.energy/len(row.numbers)
                                # print(aaa,bbb,ccc)
                                energyEndPoints[0] = row.energy/len(row.numbers)
                            elif endPointTester == formationEBasis[1]:
                                # aaa = row.energy
                                # bbb = len(row.numbers)
                                # ccc = row.energy/len(row.numbers)
                                # print(aaa,bbb,ccc)
                                energyEndPoints[1] = row.energy/len(row.numbers)

            if formationEBasis is not None:
                for key in tmpDictE:
                    tmpNum = self.get_num(key, formationEBasis[0], formationEBasis[1])
                    tmpE = tmpDictE[key] - tmpNum[0]*energyEndPoints[0] - tmpNum[1]*energyEndPoints[1]
                    tmpDictE[key] = tmpE

            self.atom_energy_data = tmpDictE
            self.atom_force_data = tmpDictF
        elif energyFile[-4:] == "json":
            with open(energyFile) as f:
                self.atom_energy_data = json.load(f)
        # self.atom_energy_data = 64
        assert self.atom_energy_data is not None, "fail to initialize self.atom_energy_data"
        # assert self.atom_force_data is not None, "fail to initialize self.atom_force_data"
        print("done init!")

    def get_num(self, input_string, A, B):
        elements = re.findall(A+'|'+B+'|\d+', input_string)
        separated_array = []

        for element in elements:
            if element.isdigit():
                separated_array.append(int(element))
            else:
                separated_array.append(element)
            
        ACount = 0
        BCount = 0
        
        for idx, i in enumerate(separated_array):
            if i == A:
                ACount += 1
            elif i == B:
                BCount += 1
            elif isinstance(i, int):
                if separated_array[idx-1] == A:
                    ACount += i-1
                elif separated_array[idx-1] == B:
                    BCount += i-1
    
        return [ACount, BCount]
        
    def energy_correction(self, formula, E):
        sperated = re.split('Li|Na', formula)
        for i in range(len(sperated)):
            if sperated[i] == '':
                sperated[i] = '1'
        if len(sperated) == 3:
            return E + int(sperated[1])*196.893 + int(sperated[2])*1297.246
        elif len(sperated) == 2:
            if formula[:2] == "Li":
                return E + int(sperated[1])*196.893
            elif formula[:2] == "Na":
                return E + int(sperated[1])*1297.246
        return np.inf


    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=['positions']):
        """Calculate the total energy of the system by summing the energies
        of individual atoms.

        Parameters:
        atoms: ASE Atoms object, optional
            The system of atoms. If not provided, the atoms object passed
            to the constructor will be used.
        properties: list of str, optional
            List of properties to calculate. Should only contain 'energy'.
        system_changes: list of str, optional
            List of changes to the system that require the calculator to
            be rerun.

        Returns:
        dict
            Dictionary of calculated properties.
        """
        self.atoms = atoms
        
        # self.results['energy'] = self.atom_energy_data
        self.results['energy'] = self.atom_energy_data[self.currAtomSymbol]
        # self.results['forces'] = self.atom_force_data[self.currAtomSymbol]
        print(self.currAtomSymbol, self.atom_energy_data[self.currAtomSymbol])

    def has_energy(self, atomSymbol):
        # return True
        if atomSymbol in self.atom_energy_data or self.byOrder:
            self.currAtomSymbol = atomSymbol
            return True
        else:
            return False



#
# # Create a system of atoms and a list of energies for each atom
# atoms = Atoms('H2O', positions=[[0, 0, 0], [0.95, 0, 0], [0, 0, 0.95]])
# energies = [1.0, 2.0, 3.0]
#
# # Create an instance of the calculator and attach it to the atoms object
# calc = EnergySumCalculator(atoms, energies)
# atoms.set_calculator(calc)
#
# # Calculate the total energy of the system
# energy = atoms.get_potential_energy()
# print(energy)  # Output: 6.0


# -*- coding: utf-8 -*-
# """Tools for interfacing with `ASE`_.
#
# .. _ASE:
#     https://wiki.fysik.dtu.dk/ase
# """
#
# import torch
# from .nn import Sequential
# import ase.neighborlist
# from . import utils
# import ase.calculators.calculator
# import ase.units
# import copy
#
#
# class Calculator(ase.calculators.calculator.Calculator):
#     """TorchANI calculator for ASE
#
#     Arguments:
#         species (:class:`collections.abc.Sequence` of :class:`str`):
#             sequence of all supported species, in order.
#         aev_computer (:class:`torchani.AEVComputer`): AEV computer.
#         model (:class:`torchani.ANIModel` or :class:`torchani.Ensemble`):
#             neural network potential models.
#         energy_shifter (:class:`torchani.EnergyShifter`): Energy shifter.
#         dtype (:class:`torchani.EnergyShifter`): data type to use,
#             by dafault ``torch.float64``.
#         overwrite (bool): After wrapping atoms into central box, whether
#             to replace the original positions stored in :class:`ase.Atoms`
#             object with the wrapped positions.
#     """
#
#     implemented_properties = ['energy', 'forces', 'stress', 'free_energy']
#
#     def __init__(self, species, aev_computer, model, energy_shifter, dtype=torch.float64, overwrite=False):
#         super(Calculator, self).__init__()
#         self.species_to_tensor = utils.ChemicalSymbolsToInts(species)
#         # aev_computer.neighborlist will be changed later, so we need a copy to
#         # make sure we do not change the original object
#         aev_computer = copy.deepcopy(aev_computer)
#         self.aev_computer = aev_computer.to(dtype)
#         self.model = copy.deepcopy(model)
#         self.energy_shifter = copy.deepcopy(energy_shifter)
#         self.overwrite = overwrite
#
#         self.device = self.aev_computer.EtaR.device
#         self.dtype = dtype
#
#         self.nn = Sequential(
#             self.model,
#             self.energy_shifter
#         ).to(dtype)
#
#     @staticmethod
#     def strain(tensor, displacement, surface_normal_axis):
#         displacement_of_tensor = torch.zeros_like(tensor)
#         for axis in range(3):
#             displacement_of_tensor[..., axis] = tensor[..., surface_normal_axis] * displacement[axis]
#         return displacement_of_tensor
#
#     def calculate(self, atoms=None, properties=['energy'],
#                   system_changes=ase.calculators.calculator.all_changes):
#         super(Calculator, self).calculate(atoms, properties, system_changes)
#         cell = torch.tensor(self.atoms.get_cell(complete=True),
#                             dtype=self.dtype, device=self.device)
#         pbc = torch.tensor(self.atoms.get_pbc(), dtype=torch.bool,
#                            device=self.device)
#         pbc_enabled = pbc.any().item()
#         species = self.species_to_tensor(self.atoms.get_chemical_symbols()).to(self.device)
#         species = species.unsqueeze(0)
#         coordinates = torch.tensor(self.atoms.get_positions())
#         coordinates = coordinates.unsqueeze(0).to(self.device).to(self.dtype) \
#                                  .requires_grad_('forces' in properties)
#
#         if pbc_enabled:
#             coordinates = utils.map2central(cell, coordinates, pbc)
#             if self.overwrite and atoms is not None:
#                 atoms.set_positions(coordinates.detach().cpu().reshape(-1, 3).numpy())
#
#         if 'stress' in properties:
#             displacements = torch.zeros(3, 3, requires_grad=True,
#                                         dtype=self.dtype, device=self.device)
#             displacement_x, displacement_y, displacement_z = displacements
#             strain_x = self.strain(coordinates, displacement_x, 0)
#             strain_y = self.strain(coordinates, displacement_y, 1)
#             strain_z = self.strain(coordinates, displacement_z, 2)
#             coordinates = coordinates + strain_x + strain_y + strain_z
#
#         if pbc_enabled:
#             if 'stress' in properties:
#                 strain_x = self.strain(cell, displacement_x, 0)
#                 strain_y = self.strain(cell, displacement_y, 1)
#                 strain_z = self.strain(cell, displacement_z, 2)
#                 cell = cell + strain_x + strain_y + strain_z
#             _, aev = self.aev_computer((species, coordinates), cell=cell, pbc=pbc)
#         else:
#             _, aev = self.aev_computer((species, coordinates))
#
#         _, energy = self.nn((species, aev))
#         energy *= ase.units.Hartree
#         self.results['energy'] = energy.item()
#         self.results['free_energy'] = energy.item()
#
#         if 'forces' in properties:
#             forces = -torch.autograd.grad(energy.squeeze(), coordinates)[0]
#             self.results['forces'] = forces.squeeze().to('cpu').numpy()
#
#         if 'stress' in properties:
#             volume = self.atoms.get_volume()
#             stress = torch.autograd.grad(energy.squeeze(), displacements)[0] / volume
#             self.results['stress'] = stress.cpu().numpy()
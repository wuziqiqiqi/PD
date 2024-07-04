from ase.build import bulk
from ase.visualize import view
from ase.io.lammpsdata import write_lammps_data

a = 3.51  # Lattice constant, you may adjust it based on your requirements

# Create the BCC lithium structure with 250 atoms
li_bcc_structure = bulk('Li', crystalstructure='bcc', a=a, cubic=True)
li_bcc_structure *= (20, 20, 20)  # Repeat the unit cell to obtain 250 atoms

# Visualize the structure (optional, you can remove this line if not needed)
view(li_bcc_structure)

# Save the structure to a file (optional)
write_lammps_data(file="/Users/Michael_wang/Desktop/Li_bcc_16000_atoms_bulk.data", atoms=li_bcc_structure, atom_style="full")
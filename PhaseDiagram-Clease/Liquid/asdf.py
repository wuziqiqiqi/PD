prepare_in = """
units          metal
atom_style     full

read_data      initial-solid.data
velocity       all create 300.0 1312343 dist gaussian

pair_style     chgnet/gpu 7/ocean/projects/cts18002lp/wuziqi/clease/lammps-ASC/potentials/CHGNET
pair_coeff     * * NPtrj-efsm   Cl K Li

variable myZlo equal zlo
variable myZhi equal zhi
variable myMid equal (myZlo + myZhi)/2

region   solid INF INF INF INF myZlo myMid
region   liquid INF INF INF INF myMid myZhi
group    g_solid region solid
group    g_liquid region liquid

neighbor        0.3 bin
neigh_modify    delay 10

timestep        0.001
compute         msd all msd
thermo_style    custom step temp pe etotal press vol c_msd[4]
thermo          10 
dump            1 all custom 10 xyz-npt.dump type id x y z
#dump           2 all custom 10 force-npt.dump type id fx fy fz

fix             1 all npt temp 300.0 300.0 $(100.0*dt) aniso 1.01325 1.01325 $(1000.0*dt)
run             5000
unfix           1

fix             2 g_liquid npt temp 300 2000 $(100.0*dt) z 1.01325 1.01325 $(1000.0*dt) couple none
run             5000
unfix           2

write_data      structure.ready.data

write_restart   structure.ready
"""

with open("prepare.in", "w") as file:
    file.write(prepare_in)
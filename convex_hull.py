from ase.db import connect
from ase.io import read, write
import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.spatial import ConvexHull

from chgnet.model.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator   
from ase.calculators.lammpslib import LAMMPSlib
from mace.calculators import MACECalculator, mace_mp
from mattersim.forcefield import MatterSimCalculator

# chgnet = CHGNet.from_file("/nfs/turbo/coe-venkvis/ziqiw-turbo/fine-tune/CHGNet/06-30-2024/bestE_epoch571_e1_f8_s9_mNA.pth.tar")
chgnet = CHGNet.load()
mattersimModel = "MatterSim-v1.0.0-1M.pth"

ASElammps = LAMMPSlib(
  lmpcmds=["pair_style meam",
            "pair_coeff * * MEAM/library.meam Li Mg MEAM/LiMg.meam Li Mg"],
  atom_types={"Li":1, "Mg":2},
  keep_alive=True,
  log_file="../LAMMPSLog.log"
)

DEVICE = 'cuda'

myASECalc = ASElammps
myCHGCalc = CHGNetCalculator(model=chgnet)
myMSCalc = MatterSimCalculator(load_path=mattersimModel, device=DEVICE)
myMACECalc = mace_mp(model="medium", dispersion=False, default_dtype="float32", device=DEVICE)
myMACEftCalc = MACECalculator(model_paths=['/nfs/turbo/coe-venkvis/ziqiw-turbo/fine-tune/MACE/10-10-2024/mace_fine_tunning_LiMg_swa.model'], device='cuda', default_dtype="float32")

# LiDFT = None
# Mg = None

data = read("/nfs/turbo/coe-venkvis/ziqiw-turbo/fine-tune/dataset/LiMg-BCCHCP-train-stress-chg.extxyz", index=":")

# for d in data:
#     if np.all(d.numbers == 3) and len(d.numbers) > 1:
#         print(f"Li: {d.get_potential_energy()}")
#         if LiDFT is None or prevLiDFTE > d.get_potential_energy():
#             LiDFT = d.copy()
#             prevLiDFTE = d.get_potential_energy()
#     elif np.all(d.numbers == 12) and len(d.numbers) > 1:
#         print(f"Mg: {d.get_potential_energy()}")
#         if Mg is None or prevMgE > d.get_potential_energy():
#             Mg = d.copy()
#             prevMgE = d.get_potential_energy()
            
            
# # LiDFT.calc = myASECalc
# # Mg.calc = myASECalc
# print()
# # print(f"Li min: {Li.get_potential_energy()}")
# # print(f"Mg min: {Mg.get_potential_energy()}")
# print(f"LiDFT min: {prevLiDFTE}")
# print(f"MgDFT min: {prevMgE}")
        
# bccDB = connect("/nfs/turbo/coe-venkvis/ziqiw-turbo/DFT/LiMg/LiMg-Jun27-bcc.db")
# for idx, bccRaw in enumerate(bccDB.select("")):
#     bcc = bccRaw.toatoms()
#     if np.all(bcc.numbers == 3) and len(bcc.numbers) > 1:
#         prevLiE = bcc.get_potential_energy()
#         print(f"Li: {bcc.get_potential_energy()}")
#         if Li is None or prevLiE > bcc.get_potential_energy():
#             Li = bcc.copy()
#             prevLiE = bcc.get_potential_energy()
#     elif np.all(bcc.numbers == 12) and len(bcc.numbers) > 1:
#         prevMgE = bcc.get_potential_energy()
#         print(f"Mg: {bcc.get_potential_energy()}")
#         if Mg is None or prevMgE > bcc.get_potential_energy():
#             Mg = bcc.copy()
#             prevMgE = bcc.get_potential_energy()
    
# calcList = [myASECalc, myCHGCalc, myMSCalc, myMACECalc, myMACEftCalc]    
# calcList = ["DFT", "myASECalc"]
calcList = ["DFT", "myASECalc", "myCHGCalc", "myMSCalc", "myMACECalc", "myMACEftCalc"]
    
E = np.zeros((len(calcList), len(data)))
X = np.zeros((len(calcList), len(data)))
N = np.zeros((len(calcList), len(data)))
endPoints = np.zeros((len(calcList), 2))


for calcIdx, myCalc in enumerate(calcList):
    Litmp = None
    Mgtmp = None
    for idx, atoms in enumerate(data):
        if myCalc != "DFT":
            atoms.calc = eval(myCalc)
        E[calcIdx, idx] = atoms.get_potential_energy()
        X[calcIdx, idx] = np.sum(atoms.numbers == 3)/len(atoms)
        N[calcIdx, idx] = len(atoms)-np.sum(atoms.numbers == 3)
        if np.all(atoms.numbers == 3) and len(atoms.numbers) > 1:
            print(f"Li: {atoms.get_potential_energy()}")
            if Litmp is None or endPoints[calcIdx, 0] > atoms.get_potential_energy():
                Litmp = atoms.copy()
                endPoints[calcIdx, 0] = atoms.get_potential_energy()
        elif np.all(atoms.numbers == 12) and len(atoms.numbers) > 1:
            print(f"Mg: {atoms.get_potential_energy()}")
            if Mgtmp is None or endPoints[calcIdx, 1] > atoms.get_potential_energy():
                Mgtmp = atoms.copy()
                endPoints[calcIdx, 1] = atoms.get_potential_energy()
        
    print()
    # print(f"Li min: {Li.get_potential_energy()}")
    # print(f"Mg min: {Mg.get_potential_energy()}")
    print(f"Li{myCalc} min: {endPoints[calcIdx, 0]}")
    print(f"Mg{myCalc} min: {endPoints[calcIdx, 1]}")
    
    # if calcIdx > 0:
    #     a = np.array([[endPoints[calcIdx, 0], 1], [endPoints[calcIdx, 1], 1]])
    #     b = np.array([endPoints[0, 0], endPoints[0, 1]])
    #     sol = np.linalg.solve(a, b)
    #     print(sol)
    #     E[calcIdx] = E[calcIdx] * sol[0] + sol[1]
    #     endPoints[calcIdx] = endPoints[calcIdx] * sol[0] + sol[1]
    
    
    # goodSimplices = []
    # for idx, simp in enumerate(hullSimplices):
    #     print(f"simp: {hullSimplices[idx]}")
    #     if formE[simp][1] <= 0:
    #         goodSimplices.append(formE[simp])
    # goodSimplices = np.array(goodSimplices)

    plt.figure()
    formE = E[calcIdx] - X[calcIdx] * endPoints[calcIdx, 0] - (1 - X[calcIdx]) * endPoints[calcIdx, 1]
    points = np.array([X[calcIdx], formE]).T
    hullSimplices = ConvexHull(points).simplices
    for simplex in hullSimplices:
        if points[simplex, 1][0] <= 0 and points[simplex, 1][1] <= 0:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
    
    formE0 = E[0] - X[0] * endPoints[0, 0] - (1 - X[0]) * endPoints[0, 1]
    points0 = np.array([X[0], formE0]).T
    hullSimplices0 = ConvexHull(points0).simplices
    for simplex in hullSimplices0:
        if points0[simplex, 1][0] <= 0 and points0[simplex, 1][1] <= 0:
            plt.plot(points0[simplex, 0], points0[simplex, 1], 'k-')
    
    plt.plot(X[0], E[0] - X[0] * endPoints[0, 0] - (1 - X[0]) * endPoints[0, 1], '.', label="DFT")
    plt.plot(X[calcIdx], formE, '.', label=myCalc)
    plt.legend()
    plt.title(f"{myCalc} vs DFT")
    plt.xlabel("X (\%Li)")
    plt.ylabel("Formation Energy")
    plt.savefig(f"{myCalc}.png")

exit()

bccDB = connect("/nfs/turbo/coe-venkvis/ziqiw-turbo/DFT/LiMg/LiMg-Jun27-bcc.db")

if os.path.isfile("bccE.npy"):
    bccE = np.load("bccE.npy", allow_pickle=True)
    bccX = np.load("bccX.npy", allow_pickle=True)
    bccN = np.load("bccN.npy", allow_pickle=True)
else:
    bccE = []
    bccX = []
    bccN = []
    for idx, bcc in enumerate(bccDB.select("")):
        bccAtoms = bcc.toatoms()

        if len(bccAtoms) == 1:
            continue
        if np.sum(bccAtoms.numbers == 3) <= len(bccAtoms)/2:
            bccAtoms.calc = myCalc
            bccXTmp = np.sum(bccAtoms.numbers == 3)/(len(bccAtoms)-np.sum(bccAtoms.numbers == 3))
            bccX.append(bccXTmp)
            bccE.append(bccAtoms.get_potential_energy())
            bccN.append(len(bccAtoms)-np.sum(bccAtoms.numbers == 3))
            print(idx, bccX[-1], bccE[-1])
        if len(bccAtoms) == np.sum(bccAtoms.numbers == 3):
            bccAtoms.calc = myCalc
            bccLiGS = bccAtoms.get_potential_energy()/len(bccAtoms)
        if np.sum(bccAtoms.numbers == 3) == 0:
            bccAtoms.calc = myCalc
            bccMgGS = bccAtoms.get_potential_energy()/len(bccAtoms)


    bccE = np.array(bccE)
    bccX = np.array(bccX)
    bccN = np.array(bccN)

    # np.save("bccE.npy", bccE)
    np.save("bccX.npy", bccX)
    np.save("bccN.npy", bccN)
    

bccEPA = bccE/bccN


hcpDB = connect("/nfs/turbo/coe-venkvis/ziqiw-turbo/DFT/LiMg/LiMg-Jun27-hcp.db")

if os.path.isfile("hcpE.npy"):
    hcpE = np.load("hcpE.npy", allow_pickle=True)
    hcpX = np.load("hcpX.npy", allow_pickle=True)
    hcpN = np.load("hcpN.npy", allow_pickle=True)
else:
    hcpE = []
    hcpX = []
    hcpN = []
    for idx, hcp in enumerate(hcpDB.select("")):
        hcpAtoms = hcp.toatoms()
        
        if len(hcpAtoms) == 1:
            continue
        if np.sum(hcpAtoms.numbers == 3) < len(hcpAtoms)/2:
            hcpAtoms.calc = myCalc
            hcpXTmp = np.sum(hcpAtoms.numbers == 3)/(len(hcpAtoms)-np.sum(hcpAtoms.numbers == 3))
            hcpX.append(hcpXTmp)
            hcpE.append(hcpAtoms.get_potential_energy())
            hcpN.append(len(hcpAtoms)-np.sum(hcpAtoms.numbers == 3))
            print(idx, hcpX[-1], hcpE[-1])
        if np.sum(hcpAtoms.numbers == 3) == 0:
            hcpAtoms.calc = myCalc
            hcpMgGS = hcpAtoms.get_potential_energy()/len(hcpAtoms)
        if np.sum(hcpAtoms.numbers == 3) == len(hcpAtoms):
            hcpAtoms.calc = myCalc
            hcpLiGS = hcpAtoms.get_potential_energy()/len(hcpAtoms)

    hcpE = np.array(hcpE)
    hcpX = np.array(hcpX)
    hcpN = np.array(hcpN)

    # np.save("hcpE.npy", hcpE)
    np.save("hcpX.npy", hcpX)
    np.save("hcpN.npy", hcpN)
    
hcpEPA = hcpE/hcpN
# hcpGS = bccEPA[bccX == 0][0]
# bccGS = hcpEPA[hcpX == 1][0]

bccFormationEPA = bccEPA -  hcpMgGS - bccX * bccLiGS
hcpFormationEPA = hcpEPA -  hcpMgGS - hcpX * bccLiGS

# print("bcc li:", bccGS, "hcp li:", hcpEPA[hcpX == 1][0])
# print("bcc mg:", bccEPA[bccX == 0][0], "hcp mg:", hcpGS)
# print("bccGS", bccGS, "hcpGS", hcpGS)

# plt.plot(bccX, bccFormationEPA, 'o')
plt.plot(hcpX, hcpFormationEPA, 'o')
# plt.xlim([0, 1])
# plt.ylim([np.min(bccFormationEPA[bccX <= 1]), np.max(bccFormationEPA[bccX <= 1])])
# plt.legend(["bcc", "hcp"])
# plt.title("HCP LiMg")
# plt.savefig("convex_hull.png")
plt.show()
    

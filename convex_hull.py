from chgnet.model.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator    

from ase.db import connect
import numpy as np
from matplotlib import pyplot as plt
import os

# chgnet = CHGNet.from_file("/nfs/turbo/coe-venkvis/ziqiw-turbo/fine-tune/CHGNet/06-30-2024/bestE_epoch571_e1_f8_s9_mNA.pth.tar")
chgnet = CHGNet.load()
chgnetCalc = CHGNetCalculator(model=chgnet)

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
            bccAtoms.calc = chgnetCalc
            bccXTmp = np.sum(bccAtoms.numbers == 3)/(len(bccAtoms)-np.sum(bccAtoms.numbers == 3))
            bccX.append(bccXTmp)
            bccE.append(bccAtoms.get_potential_energy())
            bccN.append(len(bccAtoms)-np.sum(bccAtoms.numbers == 3))
            print(idx, bccX[-1], bccE[-1])
        if len(bccAtoms) == np.sum(bccAtoms.numbers == 3):
            bccAtoms.calc = chgnetCalc
            bccLiGS = bccAtoms.get_potential_energy()/len(bccAtoms)
        if np.sum(bccAtoms.numbers == 3) == 0:
            bccAtoms.calc = chgnetCalc
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
            hcpAtoms.calc = chgnetCalc
            hcpXTmp = np.sum(hcpAtoms.numbers == 3)/(len(hcpAtoms)-np.sum(hcpAtoms.numbers == 3))
            hcpX.append(hcpXTmp)
            hcpE.append(hcpAtoms.get_potential_energy())
            hcpN.append(len(hcpAtoms)-np.sum(hcpAtoms.numbers == 3))
            print(idx, hcpX[-1], hcpE[-1])
        if np.sum(hcpAtoms.numbers == 3) == 0:
            hcpAtoms.calc = chgnetCalc
            hcpMgGS = hcpAtoms.get_potential_energy()/len(hcpAtoms)
        if np.sum(hcpAtoms.numbers == 3) == len(hcpAtoms):
            hcpAtoms.calc = chgnetCalc
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
    

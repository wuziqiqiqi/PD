import os
import numpy as np
import json

def summarizeTables(workingDir):
    oldDir = os.getcwd()
    os.chdir(workingDir)
    print(f"workingDir: {workingDir}")

    options = json.load(open('input.json'))

    for whichTable in ["phi", "X"]:

        gsNames = ['Li', 'Mg']

        muInit, muFinal = options["EMC"]["muInit"], options["EMC"]["muFinal"]
        dMu = options["EMC"]["dMu"]
        dMu = abs(dMu)
        if (muFinal - muInit)*dMu < 0:
            dMu = -dMu
        mus = np.arange(muInit, muFinal+dMu*1e-5, dMu)
        
        TInit, TFinal = options["EMC"]["TInit"], options["EMC"]["TFinal"]
        dT = options["EMC"]["dT"]
        Ts = np.arange(TInit, TFinal+1e-5, dT)
        # Ts = [800, 835, 870]
        
        print(mus)
        print(Ts)

        for gs in gsNames:
            print(gs)
            #load all file names with for fomrat gs-*-{whichTable}Table.npy
            fileNames = []
            for file in os.listdir('.'):
                if file.startswith(f"{gs}-") and file.endswith(f'-{whichTable}Table.npy'):
                    fileNames.append(file)
            fileNames.sort()
            print(fileNames)
            if len(fileNames) != len(Ts):
                print(f"Skipping {whichTable}table for {gs} in directory {workingDir}")
                # os.chdir(oldDir)
                continue
            
            theTable = np.zeros((len(Ts), len(mus)))
            for idx, file in enumerate(fileNames):
                data = np.load(file)
                theTable[idx] = data
            
                # remove file
                # os.remove(file)
            
            np.save(f'{gs}-{whichTable}Table-all.npy', theTable)
    os.chdir(oldDir)


if __name__ == "__main__":
    workingDir = os.sys.argv[1]
    summarizeTables(workingDir)
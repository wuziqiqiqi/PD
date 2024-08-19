import os
import numpy as np

# load first command line argument to determine which table to make
whichTable = os.sys.argv[1]

gsNames = ['Li', 'Mg']

mus = np.arange(-0.4, 0.4+1e-9, 0.04)
Ts = np.arange(100, 800+1e-9, 35)

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
    
    theTable = np.zeros((len(Ts), len(mus)))
    for idx, file in enumerate(fileNames):
        data = np.load(file)
        theTable[idx] = data
    
        # remove file
        os.remove(file)
    
    np.save(f'{gs}-{whichTable}Table-all.npy', theTable)
        
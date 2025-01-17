import os
import numpy as np
import json
import argparse
from phiNTTableMaker import summarizeTables

parser = argparse.ArgumentParser(description="Process some command-line arguments.")

# Add the arguments
parser.add_argument('-i', '--input', type=str, required=True, help='input yaml filename')
parser.add_argument('-n', type=str, default='3', help='number of committee members (default: 3)')
parser.add_argument('-m', '--mode', type=str, required=True, help='set mode: prep or run or clean')

# Parse the arguments
args = parser.parse_args()

assert os.path.exists(args.input), f'{args.input} does not exist'

inputFileName = args.input[:-5]

if args.mode == 'prep':
    for i in range(int(args.n)):
        if i == 0:
            os.system(f'sbatch runInit0.sh')
            os.system(f'sbatch runInit1.sh')
            continue
        os.system(f'cp {inputFileName}.yaml {inputFileName}-{i}.yaml')
        os.system(f'cp runInit0.sh runInit0-{i}.sh')
        os.system(f'cp runInit1.sh runInit1-{i}.sh')
        # add -{i} after "{inputFileName}" in file runInit-{i}.sh
        os.system(f"sed -i 's/{inputFileName}/{inputFileName}-{i}/g' runInit0-{i}.sh")
        os.system(f"sed -i 's/{inputFileName}/{inputFileName}-{i}/g' runInit1-{i}.sh")
        os.system(f'sbatch runInit0-{i}.sh')
        os.system(f'sbatch runInit1-{i}.sh')
elif args.mode == 'run':
    for i in range(int(args.n)):
        if i == 0:
            os.system(f'sbatch runBatch-{inputFileName}-gs\[0\].sh')
            os.system(f'sbatch runBatch-{inputFileName}-gs\[1\].sh')
            continue
        os.system(f'sbatch runBatch-{inputFileName}-{i}-gs\[0\].sh')
        os.system(f'sbatch runBatch-{inputFileName}-{i}-gs\[1\].sh')
elif args.mode == 'clean':
    # list all directories with format inputFileName*
    dirs = []
    for dir in os.listdir('.'):
        if dir.startswith(inputFileName) and os.path.isdir(dir):
            dirs.append(dir)
    dirs.sort()
    print(dirs)
    
    LiphiTables = []
    MgphiTables = []
    LiXTables = []
    MgXTables = []
    for dir in dirs:
        summarizeTables(dir)
        LiphiTables.append(np.load(f'{dir}/Li-phiTable-all.npy'))
        MgphiTables.append(np.load(f'{dir}/Mg-phiTable-all.npy'))
        LiXTables.append(np.load(f'{dir}/Li-XTable-all.npy'))
        MgXTables.append(np.load(f'{dir}/Mg-XTable-all.npy'))
    
    LiphiTables = np.array(LiphiTables)
    print(LiphiTables.shape)
    
    LiphiTable = np.mean(LiphiTables, axis=0)
    MgphiTable = np.mean(MgphiTables, axis=0)
    LiXTable = np.mean(LiXTables, axis=0)
    MgXTable = np.mean(MgXTables, axis=0)
    
    np.save('averaged-MACEft-Li-phiTable-all.npy', LiphiTable)
    np.save('averaged-MACEft-Mg-phiTable-all.npy', MgphiTable)
    np.save('averaged-MACEft-Li-XTable-all.npy', LiXTable)
    np.save('averaged-MACEft-Mg-XTable-all.npy', MgXTable)
    
    for dir in os.listdir('.'):
        if f"{inputFileName}-" in dir and os.path.isfile(dir):
            print(f'removing {dir}')
            # os.remove(dir)
        
        if "runInit0-" in dir and os.path.isfile(dir):
            print(f'removing {dir}')
            # os.remove(dir)
        
        if "runInit1-" in dir and os.path.isfile(dir):
            print(f'removing {dir}')
            # os.remove(dir)
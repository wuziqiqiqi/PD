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
            os.system(f'sbatch runInit.sh')
            continue
        os.system(f'cp {inputFileName}.yaml {inputFileName}-{i}.yaml')
        os.system(f'cp runInit.sh runInit-{i}.sh')
        # add -{i} after "{inputFileName}" in file runInit-{i}.sh
        os.system(f"sed -i 's/{inputFileName}/{inputFileName}-{i}/g' runInit-{i}.sh")
        os.system(f'sbatch runInit-{i}.sh')
elif args.mode == 'run':
    for i in range(int(args.n)):
        if i == 0:
            os.system(f'sbatch runBatch-{inputFileName}.sh')
            continue
        os.system(f'sbatch runBatch-{inputFileName}-{i}.sh')
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
    
    np.save('Li-phiTable-averaged.npy', LiphiTable)
    np.save('Mg-phiTable-averaged.npy', MgphiTable)
    np.save('Li-XTable-averaged.npy', LiXTable)
    np.save('Mg-XTable-averaged.npy', MgXTable)
    
    for dir in os.listdir('.'):
        if f"{inputFileName}-" in dir and os.path.isfile(dir):
            print(f'removing {dir}')
            os.remove(dir)
        
        if "runInit-" in dir and os.path.isfile(dir):
            print(f'removing {dir}')
            os.remove(dir)
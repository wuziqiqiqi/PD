import os
import sys
from mpi4py import MPI
from ase.io import read, write
from ase.data import atomic_numbers

def find_files(root_dir, filename_pattern):
    """Recursively find all files matching the filename pattern in directory."""
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == filename_pattern:
                yield os.path.join(dirpath, filename)

def read_mapping_file(file_path):
    """Reads the type mapping file and returns a dictionary of type mappings."""
    with open(file_path, 'r') as file:
        mapping_str = file.read().strip()
    if mapping_str.split()[0].isnumeric():
        mapping = list(map(int, mapping_str.split()))
    else:
        mapping = [atomic_numbers[x] for x in mapping_str.split()]
    return {i + 1: mapping[i] for i in range(len(mapping))}

def get_type_mapping_file(dir_path):
    """Attempt to read the mapping file from the current or one level above directory."""
    possible_paths = [dir_path, os.path.dirname(dir_path)]
    for path in possible_paths:
        file_path = os.path.join(path, '123toABC.txt')
        if os.path.exists(file_path):
            return file_path
    return None

def modify_and_save_atoms(file_path, type_mapping, comm, rank):
    """Modify the atom types in the file and save the new data."""
    print("reading...")
    atoms = read(file_path, index=":", parallel=False)
    # Update atom types based on the mapping
    print("changing type...")
    for atom in atoms:
        for t in type_mapping.keys():
            atom.numbers[atom.numbers == t] = type_mapping[t]
    
    new_file_path = file_path.replace('.dump', '.xyz')
    print(f"Rank {rank}: writing to {new_file_path}")
    write(new_file_path, atoms, parallel=False)
    os.remove(file_path)
    print(f"Rank {rank}: done")

    # # Serialize file writing across ranks
    # for i in range(comm.size):
    #     if i == rank:
    #         new_file_path = file_path.replace('.dump', '.xyz')
    #         print(f"Rank {rank}: writing to {new_file_path}")
    #         write(new_file_path, atoms)
    #         os.remove(file_path)
    #         print(f"Rank {rank}: done")
    #         # Signal next rank to proceed
    #         comm.bcast(True, root=i)
    #     else:
    #         # Wait for the signal from the current rank
    #         comm.bcast(None, root=i)


def main(filename_pattern):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    root_dir = '.'
    dump_files = list(find_files(root_dir, filename_pattern))

    # Divide files among processes
    files_per_process = len(dump_files) // size
    if rank < len(dump_files) % size:
        files_per_process += 1
    start_index = sum(len(dump_files) // size + (1 if i < len(dump_files) % size else 0) for i in range(rank))
    end_index = start_index + files_per_process

    local_files = dump_files[start_index:end_index]

    for dump_file in local_files:
        print(f"Rank {rank} is processing {dump_file}")
        mapping_file_path = get_type_mapping_file(os.path.dirname(dump_file))
        if mapping_file_path is None:
            print(f"No type mapping file found for {dump_file}. Skipping.")
            continue
        type_mapping = read_mapping_file(mapping_file_path)
        modify_and_save_atoms(dump_file, type_mapping, comm, rank)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename_pattern = sys.argv[1]
    else:
        print("Usage: python script.py <filename-pattern>")
        sys.exit(1)
    main(filename_pattern)

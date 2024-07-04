import os
from ase.io import read, write

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
    mapping = list(map(int, mapping_str.split()))
    return {i + 1: mapping[i] for i in range(len(mapping))}

def get_type_mapping_file(dir_path):
    """Attempt to read the mapping file from the current or one level above directory."""
    possible_paths = [dir_path, os.path.dirname(dir_path)]
    for path in possible_paths:
        file_path = os.path.join(path, '123toABC.txt')
        if os.path.exists(file_path):
            return file_path
    return None

def modify_and_save_atoms(file_path, type_mapping):
    """Modify the atom types in the file and save the new data."""
    atoms = read(file_path, index=":")
    # Update atom types based on the mapping
    for atom in atoms:
        for t in type_mapping.keys():
            atom.numbers[atom.numbers == t] = type_mapping[t]

    # Write the modified atoms to a new .xyz file
    new_file_path = file_path.replace('mix.dump', 'mix.xyz')
    write(new_file_path, atoms)
    # Delete the original file
    os.remove(file_path)

def main():
    root_dir = '.'
    dump_files = list(find_files(root_dir, 'xyz-mix.dump'))

    for dump_file in dump_files:
        mapping_file_path = get_type_mapping_file(os.path.dirname(dump_file))
        if mapping_file_path is None:
            print(f"No type mapping file found for {dump_file}. Skipping.")
            continue
        type_mapping = read_mapping_file(mapping_file_path)
        modify_and_save_atoms(dump_file, type_mapping)

if __name__ == '__main__':
    main()

import numpy as np


def name_clusters(fingerprints):
    """Name clusters based on its size and diameter."""
    s = size(fingerprints[0])
    distance_list = [2*np.sqrt(fp[0]) for fp in fingerprints]

    distance_list = np.round(distance_list, decimals=6)
    distance_list = np.unique(distance_list)
    distance_list.sort()
    prefixes = []
    prefix_dict = {}
    for i, fp in enumerate(fingerprints):
        match = get_first_match(fp, fingerprints[:i])
        if match != -1:
            prefix = prefixes[match]
        else:
            dia = diameter(fp)
            index = np.argmin(np.abs(distance_list - dia))
            assert abs(dia - distance_list[index]) < 2E-6

            if index < 10:
                prefix = 'c{}_d000{}'.format(s, index)
            elif index < 100:
                prefix = 'c{}_d00{}'.format(s, index)
            elif index < 1000:
                prefix = 'c{}_d0{}'.format(s, index)
            else:
                prefix = 'c{}_d{}'.format(s, index)
        prefixes.append(prefix)

        fp_list = prefix_dict.get(prefix, [])
        if fp not in fp_list:
            fp_list.append(fp)
        prefix_dict[prefix] = fp_list

    # Sort prefix dict
    for _, v in prefix_dict.items():
        v.sort()

    names = []
    for prefix, fp in zip(prefixes, fingerprints):
        suffix = prefix_dict[prefix].index(fp)
        names.append(prefix+'_{}'.format(suffix))
    return names


def get_first_match(fingerprint, fingerprints):
    """
    Find index of the first instance where the passed fingerprint is within the
    list of the fingerprints.
    """
    for i, fp in enumerate(fingerprints):
        if np.allclose(fp, fingerprint):
            return i
    return -1


def diameter(fingerprint):
    """Get diameter of a cluster using its fingerprint."""
    return 2 * np.sqrt(fingerprint[0])


def size(fingerprint):
    """Get size of the cluster using its fingerprint."""
    num = np.sqrt(2 * len(fingerprint) + 0.25)
    s = num - 0.5
    assert abs(int(s) - s) < 1E-9
    return int(s)


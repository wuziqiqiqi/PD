import numpy as np

def name_clusters(fingerprints):
    """Name clusters based on its size and diameter."""
    s = size(fingerprints[0])
    distance_list = [2*np.sqrt(fp[0]) for fp in fingerprints]

    distance_list = np.round(distance_list, decimals=6)
    distance_list = np.unique(distance_list)
    distance_list.sort()
    names = []
    num_occurences = {}
    for i, fp in enumerate(fingerprints):
        match = get_first_match(fp, fingerprints[:i])
        if match != -1:
            names.append(names[match])
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
            counter = 0
            if prefix not in num_occurences.keys():
                num_occurences[prefix] = 1
                counter = 0
            else:
                counter = num_occurences[prefix]
                num_occurences[prefix] += 1

            name = prefix + '_{}'.format(counter)
            names.append(name)
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


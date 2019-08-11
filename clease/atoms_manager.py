class AtomsManager(object):
    """
    Class that manages the Atoms object used in a cluster expansion
    context. This class can return indices of the Atoms object grouped
    according to various schemes.
    """
    def __init__(self, atoms):
        self.atoms = atoms

    def index_by_tag(self):
        """
        Return indices by tag. Assumes that the tags
        is a continuous sequence starting from 0.
        """
        tags = set()
        for atom in self.atoms:
            tags.add(atom.tag)

        tags = sorted(list(tags))
        ind_by_tag = [[] for _ in tags]

        for atom in self.atoms:
            ind_by_tag[atom.tag].append(atom.index)
        return ind_by_tag

    def index_by_symbol(self, symbols):
        """
        Return index by groups defined by its symbols

        Example:
            If symbols = ['Au', 'Cu'] then indices where with Au
            is returned as group 1 and indices with Cu as group 2

            If symbols = ['Au, ['Cu', 'X'], 'Ag'] then indices with Au is
            returnes as group 1 and indices with Cu OR X is returned
            as group 2 and Ag as group 3

        Parameters:
        symbols: list
            List with symbols that define a group
        """
        ind_by_symbol = [[] for _ in symbols]
        group_map = {}
        for i, item in enumerate(symbols):
            if isinstance(item, list):
                for x in item:
                    group_map[x] = i
            else:
                group_map[item] = i

        # Loop over indices
        for atom in self.atoms:
            ind_by_symbol[group_map[atom.symbol]].append(atom.index)
        return ind_by_symbol

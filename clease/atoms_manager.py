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

    def unique_elements(self, ignore=[]):
        """
        Return a list with unique elements
        """
        all_unique = set([a.symbol for a in self.atoms])
        return list(all_unique - set(ignore))

    def single_element_sites(self, allowed_elements):
        """
        Return a list of sites that can only occupied by a single
        element according to allowed_elements.

        Parameters:

        allowed_elements: list
            List with allowed elements on each site. It is assumed
            that all elements first in each group is present in self.atoms
            If allowed_elements is equal to [['Au', 'Ag' 'X], ['Cu', 'X']] it
            means that all sites where `self.atoms` has a gold symbol can be
            occupied by Au, Ag, X in the cluster expansion and all sites that
            are occupied by Cu can be occubpied by Cu or X in the cluster
            expansion.
        """
        single_site_symb = [x[0] for x in allowed_elements if len(x) == 1]
        single_sites = []
        for atom in self.atoms:
            if atom.symbol in single_site_symb:
                single_sites.append(atom.index)
        return single_sites

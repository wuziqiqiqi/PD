class BasisSpecifiedInManyGroupsError(ValueError):
    pass


def parse_max_cluster_dia(string):
    """Parse max cluster dia."""
    string = string.strip('[()]')
    try:
        diameter = float(string)
        return diameter
    except Exception:
        pass

    try:
        diameter = [float(x) for x in string.split(',')]
        return diameter
    except Exception:
        pass

    msg = "Max cluster dia has to be a float or a comma separated list where "
    msg += "each number \nis a cutoff of for 2-body clusters and beyond "
    msg += "(i.e., 3.0, 4.0, 2.0 when the max cluster size is 4)"
    raise ValueError(msg)


def parse_grouped_basis_elements(string):
    """Parse grouped basis argument."""
    string = string.replace('[', '(')
    string = string.replace(']', ')')
    string = string.replace('(', '')
    split = string.split('),')
    split = [x.replace(')', '') for x in split]
    exc_type = ValueError
    msg = 'Grouped basis has to be a list (0, 1, 2, 3) or (0, 1), (2, 3)'
    try:
        grouped = []
        for item in split:
            grouped.append([int(x) for x in item.split(',')])

        # Check that we have only unique sites
        index_set = set()
        index_list = list()
        for grp in grouped:
            index_set = index_set.union(grp)
            index_list += grp

        if len(index_set) != len(index_list):
            exc_type = BasisSpecifiedInManyGroupsError
            msg = 'Basis specified in multiple groups!'
            raise exc_type(msg)
        return grouped

    except Exception:
        raise exc_type(msg)


def parse_size(string):
    """Parse size argument."""
    # List of 3 numbers
    try:
        size = [int(x) for x in string.split(',')]
        if len(size) != 3:
            raise ValueError()
        return size
    except Exception:
        pass

    # Try 3x3 matrix
    string = string.replace('[', '(')
    string = string.replace(']', ')')
    filtered = string.replace('(', '')
    splitted = filtered.split('),')
    splitted = [x.replace(')', '') for x in splitted]
    try:
        matrix = []
        for x in splitted:
            row = [int(y) for y in x.split(',')]
            if len(row) != 3:
                raise ValueError()
            matrix.append(row)

        if len(matrix) == 1:
            matrix = matrix[0]

        if len(matrix) != 3:
            raise ValueError()

        return matrix

    except Exception:
        pass

    msg = 'Size has to be a list with 3 ints (i.e. 3, 4, 5) or rows of a '
    msg += '3x3 matrix \n(1, 0, -1), (0, 1, 1), (2, 0, 3)'
    raise ValueError(msg)


def parse_elements(string):
    """Parse elements given by user."""
    from ase.data import chemical_symbols
    string = string.replace('[', '(')
    string = string.replace(']', ')')
    try:
        if string.count(')') == 0:
            # Single basis
            symb_list = string.split(',')
            symb_list = [x.strip() for x in symb_list]

            if any(x not in chemical_symbols for x in symb_list):
                raise ValueError()
            return [symb_list]

        filtered = string.replace('(', '')
        filtered = filtered.split('),')
        symb_list = [[x.replace(')', '').strip() for x in sub.split(',')] for sub in filtered]

        for sub in symb_list:
            if any(x not in chemical_symbols for x in sub):
                raise ValueError()
        return symb_list
    except Exception:
        pass

    msg = 'Elements must be a comma-separated chemical symbols '
    msg += '(i.e. Al, Mg, Cu) for single sublattice\nand a list '
    msg += 'of symbols for each sublattice otherwise (i.e. (Al, Mg)'
    msg += ', (Zn, Cu), (X, Zn, Cu))'
    raise ValueError(msg)


def parse_cellpar(string):
    """Parse the cell parameter string."""
    string = string.strip('[()]')
    try:
        cellpar = [float(x) for x in string.split(',')]
        if len(cellpar) != 6:
            raise ValueError()
        return cellpar

    except Exception:
        pass

    msg = 'Cell parameters has to be a list of six numbers (a, b, c, alpha, '
    msg += 'beta, gamma). See ASE doc.'
    raise ValueError(msg)


def parse_cell(string):
    """Parse the cell string."""
    string = string.replace('[', '(')
    string = string.replace(']', ')')

    filtered = string.replace('(', '')
    filtered = filtered.split('),')
    filtered = [[x.replace(')', '').strip() for x in sub.split(',')] for sub in filtered]

    cell = []
    try:
        for sub in filtered:
            row = [float(x) for x in sub]
            if len(row) != 3:
                raise ValueError()
            cell.append(row)

        if len(cell) != 3:
            raise ValueError()
        return cell
    except Exception:
        pass

    msg = 'Cell has to be row of a 3x3 matrix (i.e (0.0, 4.0, 4.0), (2.0, 1.0,'
    msg += ' 6.0), (3.0, 5.0, 6.0)'
    raise ValueError(msg)


def parse_coordinate_basis(string):
    string = string.replace('[', '(')
    string = string.replace(']', ')')

    filtered = string.replace('(', '')
    filtered = filtered.split('),')
    filtered = [[x.replace(')', '').strip() for x in sub.split(',')] for sub in filtered]

    cell = []
    try:
        for sub in filtered:
            row = [float(x) for x in sub]
            if len(row) != 3:
                raise ValueError()
            cell.append(row)

        return cell
    except Exception:
        pass

    msg = 'Coordinate basis has to be list of coordinates (i.e (1.0, 4.0, 4.0)'
    msg += ', (2.0, 1.0, 6.0)'
    raise ValueError(msg)


def parse_temperature_list(string):
    try:
        temps = [float(x) for x in string.split(',')]
        return temps
    except Exception:
        pass
    msg = 'The temperatures has to be a comma separted list of values'
    raise ValueError(msg)


def parse_comma_sep_list_of_int(string):
    try:
        values = [int(x) for x in string.split(',')]
        return values
    except Exception:
        pass
    msg = 'Could not parse list'
    raise ValueError(msg)


def parse_concentration_list(string):
    try:
        per_basis = [x.replace('(', '').replace(')', '') for x in string.split('),')]
        conc_per_basis = []
        for b in per_basis:
            conc_per_basis.append([float(x) for x in b.split(',')])
        return conc_per_basis
    except Exception:
        pass

    msg = 'Concentration list has to be given as one comma separated list per '
    msg += 'basis'
    raise ValueError(msg)


def parse_select_cond(txt):
    """
    Parse a select condition given as a string.

    Parameter

    txt: str
        Select condition on the form gen=3,struct_type=initial
    """
    select_cond = []
    known_op = ['>=', '<=', '<', '>', '=']
    for cond in txt.split(','):
        for op in known_op:
            if op in cond:
                split = cond.split(op)
                value = split[1]
                try:
                    float_val = float(value)
                    value = float_val
                except Exception:
                    pass

                select_cond.append((split[0], op, value))
                break
    return select_cond

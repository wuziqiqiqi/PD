def parse_max_cluster_dia(string):
    """
    Parse max cluster dia
    """
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
    """
    Parse grouped basis argument
    """
    string = string.replace('(', '')
    split = string.split('),')
    split = [x.replace(')', '') for x in split]
    try:
        grouped = []
        for item in split:
            grouped.append([int(x) for x in item.split(',')])
        return grouped

    except Exception as exc:
        msg = 'Grouped basis has to be a list (0, 1, 2, 3) or (0, 1), (2, 3)'
        raise ValueError(msg)


def parse_size(string):
    """
    Parse size argument
    """
    # List of 3 numbers
    try:
        size = [int(x) for x in string.split(',')]
        if len(size) != 3:
            raise ValueError()
        return size
    except Exception:
        pass

    # Try 3x3 matrix
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

        if len(matrix) != 3:
            raise ValueError()
        return matrix

    except Exception as exc:
        pass

    msg = 'Size has to be a list with 3 ints (i.e. 3, 4, 5) or rows of a '
    msg += '3x3 matrix \n(1, 0, -1), (0, 1, 1), (2, 0, 3)'
    raise ValueError(msg)


def parse_elements(string):
    """
    Parse elements given by user
    """
    from ase.data import chemical_symbols
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
        symb_list = [[x.replace(')', '').strip() for x in sub.split(',')]
                     for sub in filtered]

        for sub in symb_list:
            if any(x not in chemical_symbols for x in sub):
                raise ValueError()
        return symb_list
    except Exception:
        pass

    msg = 'Elements has to be a list of comma separated chemical symbols '
    msg += '(i.e. Al, Mg, Cu) \nin the case of a single sublattice and a list '
    msg += 'with symbels for each sublattice \nin case of many (i.e. (Al, Mg)'
    msg += ', (Zn, Cu), (X, Zn, Cu)'
    raise ValueError(msg)


def parse_cellpar(string):
    """
    Parse the cell parameter string
    """
    try:
        cellpar = [float(x) for x in string.split(',')]
        if len(cellpar) != 6:
            raise ValueError()
        return cellpar
    except Exception as exc:
        pass

    msg = 'Cell parameters has to be a list of six numbers (a, b, c, alpha, '
    msg += 'beta, gamma). See ASE doc.'
    raise ValueError(msg)


def parse_cell(string):
    """
    Parse the cell string
    """
    filtered = string.replace('(', '')
    filtered = filtered.split('),')
    filtered = [[x.replace(')', '').strip() for x in sub.split(',')]
                for sub in filtered]

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
    filtered = string.replace('(', '')
    filtered = filtered.split('),')
    filtered = [[x.replace(')', '').strip() for x in sub.split(',')]
                for sub in filtered]

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

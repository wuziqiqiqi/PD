from ase.calculators.emt import EMT
from ase.db import connect
import sys
from ase.constraints import StrainFilter
from ase.optimize import BFGS
from clease.tools import update_db


def main(argv):
    db_id = int(argv[0])
    db_name = argv[1]

    db = connect(db_name)
    calc = EMT()
    atoms = db.get(id=db_id).toatoms()
    atoms.set_calculator(calc)
    str_filter = StrainFilter(atoms)

    relaxer = BFGS(str_filter)
    relaxer.run(fmax=0.003)
    update_db(uid_initial=db_id, final_struct=atoms,
              db_name=db_name)


if __name__ == '__main__':
    main(sys.argv[1:])

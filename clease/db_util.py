from ase.db import connect
from typing import List, Dict, Set


def get_all_cf_names(db_name: str, tab_name: str) -> Set[str]:
    """
    Return a list with all correlation function names

    Parameters:

    db_name: Name of the database

    tab_name: Name of the table in the database where correlation functions
        are stores
    """
    with connect(db_name) as db:
        query = f"SELECT key FROM {tab_name}"
        cur = db.connection.cursor()
        db_names = set()
        cur.execute(query)
        for row in cur.fetchall():
            db_names.add(row[0])
    return db_names


def get_all_cf(db_name: str, tab_name: str, db_id: int) -> Dict[str, float]:
    """
    Return all correlation functions associated with an entry in the database

    Parameters:

    db_name: Name of the database

    tab_name: Name of the table where the correlation functions are stored

    db_id: ID of the entry
    """
    with connect(db_name) as db:
        query = f"SELECT key, value FROM {tab_name} WHERE id=?"
        cur = db.connection.cursor()
        cur.execute(query, (db_id,))
        cf = {}
        for row in cur.fetchall():
            cf_name, cf_value = row
            cf[cf_name] = cf_value
    return cf


def get_cf_tables(db_name: str) -> List[str]:
    """
    Return a list with table names that contain correlation functions

    Parameter:

    db_name: Name of the database
    """
    db = connect(db_name)
    ext_tab = db._get_external_table_names()
    cf_tables = [n for n in ext_tab if n.endswith('_cf')]
    return cf_tables

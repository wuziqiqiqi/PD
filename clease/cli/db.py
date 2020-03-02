from clease.db_util import (get_all_cf_names, get_all_cf, get_cf_tables)
import argparse


def db_cli(args: argparse.Namespace):
    if args.show == 'tab':
        show_cf_tables(args.name)
    elif args.show == 'names':
        show_cf_names(args.name)
    elif args.show == 'cf':
        try:
            show_cf(args.name, args.id)
        except Exception as exc:
            print(exc)


def show_cf_tables(db_name: str):
    tables = get_cf_tables(db_name)
    cftab = ', '.join(tables)
    print(f"Available correlation function tables: {cftab}")


def show_cf_names(db_name: str):
    table = get_cf_tables(db_name)[0]
    names = sorted(get_all_cf_names(db_name, table))
    print("Tracked correlation functions")

    line = ""
    char_per_line = 110
    for name in names:
        if len(line) + len(name) > char_per_line:
            print(line)
            line = ""

        if line == "":
            line += name
        else:
            line += ', ' + name
    print(line)


def show_cf(db_name: str, db_id: int):
    table = get_cf_tables(db_name)[0]
    cfs = get_all_cf(db_name, table, db_id)
    print(f"Correlation functions for ID {db_id}")
    for k, v in cfs.items():
        print(f"{k}: {v}")

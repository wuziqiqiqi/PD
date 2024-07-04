CLEASE Command Line Interface
==============================

The CLEASE package comes with a convenient command line tools,
that can be used for various things.

1. Listing all tables in your database that contains correlation functions

    .. code-block:: console

        $ clease db mydb.db --show tab

2. Listing all the names of the correlation functions stored in your database

    .. code-block:: console

        $ clease db mydb.db --show names

3. Listing all the correlation functions of a particular entry

    .. code-block:: console

        $ clease db mydb.db --show cf --id 1

CLEASE Command Line Interface
==============================

The CLEASE package comes with a convenient command line tools, 
that can be used for various things.

1. Starting the GUI

    .. code-block:: console

        $ clease gui

2. Installing missing GUI dependencies

    .. code-block:: console

        $ clease gui --setup

3. Listing all tables in your database that contains correlation functions

    .. code-block:: console

        $ clease db mydb.db --show tab

4. Listing all the names of the correlation functions stored in your database

    .. code-block:: console

        $ clease db mydb.db --show names

5. Listing all the correlation functions of a particular entry

    .. code-block:: console

        $ clease db mydb.db --show cf --id 1
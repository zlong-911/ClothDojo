"""Test-process bootstrap for the conda-backed project Python.

Load the standard-library SQLite extension before optional CUDA packages. Some
CUDA wheels load the host libstdc++ globally; this project's conda SQLite/ICU
pair requires the newer libstdc++ shipped in the same environment.
"""

import sqlite3

assert sqlite3.sqlite_version

# SymNMF Project
SymNMF (C + Python)

This project implements Symmetric Non-negative Matrix Factorization (SymNMF) using C for efficient computations and Python for interface and execution.

Project Structure
symnmf.c – core implementation (matrix operations + SymNMF algorithm)
symnmf.h – header file
symnmfmodule.c – Python-C API wrapper
symnmf.py – main CLI program
analysis.py – optional analysis script
setup.py – build script
Makefile – alternative build method
Build

Using setup.py:

python3 setup.py build_ext --inplace

Or using Makefile:

make
Usage
python3 symnmf.py <k> <goal> <input_file>
Arguments
k – number of clusters (used only for symnmf)
goal – one of:
sym – similarity matrix
ddg – diagonal degree matrix
norm – normalized similarity matrix
symnmf – full SymNMF algorithm
input_file – path to input data file
Input Format
Each line represents a data point
Values are separated by commas or spaces
All rows must have the same number of values

Example:

1.0,2.0
3.0,4.0
5.0,6.0
Output Format
Matrices are printed row by row
Values are formatted to 4 decimal places
Values are comma-separated
Notes
Heavy computations are implemented in C for efficiency
Python handles input parsing, validation, and initialization
The SymNMF optimization runs in C via a Python extension
Error Handling

On any error, the program prints:

An Error Has Occurred

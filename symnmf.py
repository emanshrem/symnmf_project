"""
symnmf.py — Python interface for the C extension.

CLI:
    python3 symnmf.py <k> <goal> <file>

    goal ∈ {"sym", "ddg", "norm", "symnmf"}
    k is used only for "symnmf" and must satisfy 1 < k < n.

On any error, prints "An Error Has Occurred" and exits(1).
All matrices are printed with 4 decimal digits, one row per line, comma-separated.
"""

import sys
from typing import List
import numpy as np


# ----- constants -----
ERR_MSG="An Error Has Occurred"
EPS: float = 1e-4
MAX_ITER: int = 300
BETA: float = 0.5
DEN_EPS: float = 1e-6

np.random.seed(1234)



# ===== small helpers =====

def handle_error() -> None:
    """Print the single required error line and exit with non-zero code."""
    print(ERR_MSG)
    sys.exit(1)

def load_points(path: str) -> List[List[float]]:
    """Load N×d points from a text file.

    Each non-empty line is one point, with comma OR whitespace separators.
    On any parse error, empty file, or inconsistent row length: print the
    required error line and exit.
    """
    try:
        rows: List[List[float]] = []
        with open(path, "r") as f:
            for line in f:
                s = line.strip() # drop trailing/leading spaces/newlines
                if not s:        # skip blank lines (if any)
                    continue
                parts = s.replace(",", " ").split()     # split on any whitespace
                rows.append([float(x) for x in parts])  # parse each token as float

        if not rows:              # empty file → error
            handle_error()
        d = len(rows[0])          # dimension of first row
        # all rows must have the same length d
        if any(len(r) != d for r in rows):
            handle_error()

        return rows
    except Exception:
        handle_error()

def _avg(mat: List[List[float]]) -> float:
    """Average of all entries in a 2D list (used to scale H’s random init)."""
    total = 0.0
    count = 0
    for r in mat: # each row
        for v in r: # each value
            total += float(v)
            count += 1
    return (total / count) if count else 0.0 #avoid div by 0

def init_H(W: List[List[float]], k: int) -> np.ndarray:
    """
    Random non-negative initialization for H (n×k).
    Upper bound follows the course’s guideline based on avg(W) and k.
    """
    avg_W = _avg(W)
    upper = 2.0*np.sqrt(avg_W/float(k)) if k>0 else 0.0
    return np.random.uniform(0.0, upper, size=(len(W), k)) #low=0.0, high=upper, shape=n × k, where n = len(W)


#continue from here 
def print_matrix(M):
    data = M.tolist() if hasattr(M,"tolist") else M
    for r in data:
        print(",".join(f"{float(v):.4f}" for v in r))

def parse_int_like_k(raw):
    """
    Parse the command-line argument for k.
    Accepts:
        - A non-negative decimal integer (leading zeros allowed).
        - The same, optionally preceded by a plus sign.
        - A floating-point literal whose fractional part consists only of zeros
        (i.e., an integral floating-point representation), with an optional leading plus.
    Rejects:
        - Any negative value.
        - Any fractional part containing nonzero digits.
        - Empty or non-numeric input.
    Returns:
        The integer value of k if valid; otherwise None.
    """
    if not isinstance(raw, str): #sys.argv always gives strings
        return None
    s=raw.strip() #removes leading and trailing whitespace from the string.
    if not s:
        return None 

    #Handle optional leading '+'
    if s[0] == '+':
        s = s[1:]
        if not s:  #"+" alone
            return None

    if '.' in s:
        whole, frac = s.split('.', 1)
        #whole must be digits; frac must be non-empty and all zeros
        if not whole.isdigit():
            return None
        if frac=='' or any(ch != '0' for ch in frac): #we don't accept 3. for example
            return None
        return int(whole)
    else:
        if not s.isdigit():
            return None
        return int(s)

def num_rows(X):
    """Return number of rows in matrix-like object X, or None if not possible."""
    try:
        return len(X)
    except Exception:
        return None


def main():
    try:
        import symnmf_c as cmod  #C extension module

        #Expect exactly 3 args: k goal file
        if len(sys.argv) != 4:
            handle_error()

        org_k=sys.argv[1]
        goal=sys.argv[2].strip().lower()
        file_name=sys.argv[3]

        #Validate goal
        if goal not in ("sym", "ddg", "norm", "symnmf"):
            handle_error()

        #Validate and parse k if it's in one of the allowed shapes but isn't int in python 
        k=parse_int_like_k(org_k) #returns None if k is invalid
        if k is None:
            handle_error()

        #Load data
        X=load_points(file_name)
        if X is None:
            handle_error()

        n=num_rows(X)
        if n is None:
            handle_error()

        #Require 1 < k < n for ALL goals, some goals don't need k but we think it's better to be consistent to not confuse the user
        #if not (1 < k < n):
           # handle_error()

        #Implementation
        if goal=="sym":
            A=cmod.sym(X)
            print_matrix(A)

        elif goal=="ddg":
            D=cmod.ddg(X)
            print_matrix(D)

        elif goal=="norm":
            W=cmod.norm(X)
            print_matrix(W)

        elif goal=="symnmf":
            n = len(X)   # or len(points) if that's your variable name
            if not (1 < k < n):
                handle_error()  # prints "An Error Has Occurred" and exits
            W=cmod.norm(X)
            H0=init_H(W, k)  #Only H0 is calculated in Python
            Hf=cmod.symnmf(H0.tolist(), W, EPS, MAX_ITER, BETA, DEN_EPS)
            print_matrix(Hf)

        else: 
            handle_error()

    except Exception:
        handle_error()


if __name__ == "__main__":
    main()


#only main is > 40 lines, check others after documentation
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
import numpy as np


# ----- constants -----
ERR_MSG="An Error Has Occurred"
EPS: float = 1e-4
MAX_ITER: int = 300
BETA: float = 0.5
DEN_EPS: float = 1e-6

np.random.seed(1234)


# ---------------------------------------------------------------------------
# Error & IO utilities
# ---------------------------------------------------------------------------

def handle_error() :
    """Print the single required error line and exit with non-zero code."""
    print(ERR_MSG)
    sys.exit(1)

def load_points(path) :
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

def print_matrix(M) :
    """Print a matrix (2D list or NumPy array) with 4 decimal places, comma-separated."""

    # Convert NumPy-like objects to nested Python lists when possible
    data = M.tolist() if hasattr(M,"tolist") else M

    # Iterate row by row; each `row` should itself be an iterable of values.
    for r in data:
        # For each value `v` in the row:
        #   - convert to float (ensures numeric formatting works)
        #   - format with 4 decimal places via :.4f
        # Join the formatted strings with commas and print the resulting line.

        print(",".join(f"{float(v):.4f}" for v in r))



# ---------------------------------------------------------------------------
# Small math helpers
# ---------------------------------------------------------------------------

def _avg(mat) :
    """Average of all entries in a 2D list (used to scale H’s random init)."""
    total = 0.0
    count = 0
    for r in mat: # each row
        for v in r: # each value
            total += float(v)
            count += 1
    return (total / count) if count else 0.0 #avoid div by 0

def init_H(W, k) :
    """
    Random non-negative initialization for H (n×k).
    Upper bound follows the course’s guideline based on avg(W) and k.
    """
    avg_W = _avg(W)
    upper = 2.0*np.sqrt(avg_W/float(k)) if k>0 else 0.0
    return np.random.uniform(0.0, upper, size=(len(W), k)) #low=0.0, high=upper, shape=n × k, where n = len(W)



# ---------------------------------------------------------------------------
# CLI parsing & validation
# ---------------------------------------------------------------------------

def parse_int_like_k(raw) :
    """
    Parse the command-line argument for k.
    Accepts:
        - A non-negative decimal integer (leading zeros allowed).
        - The same, optionally preceded by a plus sign.
        - A floating-point literal whose fractional part consists only of zeros (i.e., an integral floating-point representation), with an optional leading plus.
    Rejects:
        - Any negative value, Any fractional part containing nonzero digits, Empty or non-numeric input.
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
        if not s.isdigit(): #isdigit() Return True if the string is a digit string(contains at least one digit), False otherwise.
            return None
        return int(s)

def _validate_k_for_symnmf(k, n) :
    """
    Enforce 1 < k < n for the 'symnmf' goal (as required by the tester).
    On violation, triggers the standard error.
    """
    if not (1 < k < n):
        handle_error()

def num_rows(X) :
    """Return number of rows in matrix-like object X, or None if not possible."""
    try:
        return len(X)
    except Exception:
        return None



# ---------------------------------------------------------------------------
# Goal dispatch
# ---------------------------------------------------------------------------

def _parse_cli(argv) :
    """
    Parse and validate CLI args.

    Expected form: symnmf.py k goal file_name
    - goal in {'sym', 'ddg', 'norm', 'symnmf'}
    - k validated with parse_int_like_k
    """
    if len(argv) != 4:
        handle_error()

    k_raw, goal_raw, file_name = argv[1], argv[2], argv[3]

    goal = goal_raw.strip().lower()
    if goal not in ("sym", "ddg", "norm", "symnmf"):
        handle_error()

    k = parse_int_like_k(k_raw)
    if k is None:
        handle_error()

    return k, goal, file_name

def _run_goal(goal, X, k) :
    """
    Compute and print the result for the requested goal using the C extension.

    For 'symnmf' we also validate k and initialize H in Python before
    handing optimization to the C extension.
    """
    import symnmf_c as cmod  # local import keeps module scope clean

    if goal == "sym":
        S = cmod.sym(X)
        print_matrix(S)
        return

    if goal == "ddg":
        D = cmod.ddg(X)
        print_matrix(D)
        return

    if goal == "norm":
        W = cmod.norm(X)
        print_matrix(W)
        return

    if goal == "symnmf":
        n = len(X)
        _validate_k_for_symnmf(k, n)
        W = cmod.norm(X)
        H0 = init_H(W, k)  # H0 (n x k) initialized in Python
        Hf = cmod.symnmf(H0.tolist(), W, EPS, MAX_ITER, BETA, DEN_EPS)
        print_matrix(Hf)
        return

    # Should be unreachable due to _parse_cli
    handle_error()



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    try:
        #Parse CLI
        k, goal, file_name = _parse_cli(sys.argv)

        #Load data
        X=load_points(file_name)
        if num_rows(X) is None:
            handle_error()

        _run_goal(goal, X, k)

    except Exception:
        handle_error()


if __name__ == "__main__":
    main()


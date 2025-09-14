import sys
import numpy as np

np.random.seed(1234)
ERR_MSG="An Error Has Occurred"
#try
EPS=1e-4
MAX_ITER=300
BETA=0.5
DEN_EPS=1e-6

def handle_error():  
    print(ERR_MSG)
    sys.exit(1)

def load_points(path): 
    try:
        pts=[]
        with open(path) as f:
            for line in f: 
                s=line.strip()
                if s:  #skip empty lines if exist, and normalize separators: turn commas into spaces, then split
                    pts.append([float(x) for x in s.replace(',', ' ').split()])
        d=len(pts[0]); 
        if not pts or any(len(r)!=d for r in pts): #in case of one of the points not in d Dimension, go to handle_error
            handle_error()
        return pts
    except Exception:
        handle_error()

def avg_w(M): #getting m for intializing H
    t=c=0.0 #t for total, c for count
    for r in M:
        for v in r: 
            t+=float(v); c+=1
    return t/c if c else 0.0 #avoid divison by zero

def init_H(W,k): 
    m = avg_w(W)
    upper = 2.0*np.sqrt(m/float(k)) if k>0 else 0.0
    return np.random.uniform(0.0, upper, size=(len(W), k)) #low=0.0, high=upper, shape=n × k, where n = len(W)

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
        import symnmf as cmod  #C extension module

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
        if not (1 < k < n):
            handle_error()

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


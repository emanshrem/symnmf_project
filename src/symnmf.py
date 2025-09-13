# symnmf.py  — Python driver (H init + printing only)
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

def main():
    """
    goal ∈ {symnmf, sym, ddg, norm}
      - symnmf: H0 is initialized here; call C's symnmf(H0, W, …) and print H
      - sym:call C's sym(X) and print A
      - ddg:call C's ddg(X)and print D
      - norm:call C's norm(X)    and print W
    """
    try:
        import symnmf as cmod  #C extension module
        # Read args
        k=int(sys.argv[1])           
        goal=sys.argv[2].strip().lower()
        file_name=sys.argv[3]
        X=load_points(file_name)

        if goal=="sym":
            A=cmod.sym(X)           
            print_matrix(A)

        elif goal == "ddg":
            D=cmod.ddg(X)            
            print_matrix(D)

        elif goal=="norm":
            W=cmod.norm(X)           
            print_matrix(W)

        elif goal=="symnmf":
            W  = cmod.norm(X)                         
            H0 = init_H(W, k) #only H0 is calculated in python                        
            Hf = cmod.symnmf(H0.tolist(), W, EPS, MAX_ITER, BETA, DEN_EPS)
            print_matrix(Hf)
            
        else:
            handle_error()

    except Exception:
        handle_error()

if __name__ == "__main__":
    main()


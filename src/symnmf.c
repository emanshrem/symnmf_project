#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <math.h> 

/* We store matrices as one flat, contiguous double[] in row-major order.
 * This needs only one allocation/free, improves CPU cache locality, and
 * makes bulk ops faster */

typedef struct { //allowed?
    int rows, cols;
    double *data;           
} Matrix;

/*error handling*/
static void handle_error(void) {
    printf("An Error Has Occurred");
    exit(1);
}

/*matrix helpers*/
Matrix create_matrix(int r, int c) {
    Matrix M;
    M.rows = r;
    M.cols = c;
    size_t n = (size_t)r * (size_t)c;
    M.data = (double*)calloc(n, sizeof(double));
    if (!M.data) handle_error();
    return M;
}
void free_matrix(Matrix *M) {
    if (M && M->data) {
        free(M->data);
        M->data = NULL; }
    M->rows = M->cols = 0;
}
double mget(const Matrix *M, int i, int j) {
    return M->data[(size_t)i * (size_t)M->cols + (size_t)j];
}
void mset(Matrix *M, int i, int j, double v) {
    M->data[(size_t)i * (size_t)M->cols + (size_t)j] = v;
}
void mult_matrix(const Matrix *A, const Matrix *B, Matrix *C) {
    int r = A->rows;
    int m = A->cols;
    int c = B->cols;

    /*sizes must match, and output shape must be right*/
    if (B->rows != m || C->rows != r || C->cols != c) {
        handle_error();
    }

    for (int i = 0; i < r; ++i) {
        for (int k = 0; k < c; ++k) {
            double s = 0.0;
            for (int j = 0; j < m; ++j) {
                s += mget(A, i, j) * mget(B, j, k);
            }
            mset(C, i, k, s);
        }
    }
}

void transpose_matrix(const Matrix *A, Matrix *AT) {
    int r = A->rows;
    int c = A->cols;

    if (AT->rows != c || AT->cols != r) {
        handle_error();
    }

    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            mset(AT, j, i, mget(A, i, j));
        }
    }
}

/* Squared Frobenius norm: ||A - B||_F^2 */
double frobenius_diff_sq(const Matrix *A, const Matrix *B) {
    int r = A->rows, c = A->cols;
    double s = 0.0;
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            double d = mget(A,i,j) - mget(B,i,j);
            s += d*d;
        }
    }
    return s;
}

/*file loading: datapoints X*/
/*Accepts commas and/or spaces as separators */
static Matrix load_points(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) handle_error();

    const char *delims = " ,\t\r\n";
    char *line = NULL;
    size_t cap = 0;

    size_t vals_cap = 1024, vals_len = 0;
    double *vals = (double*)malloc(vals_cap * sizeof(double));
    if (!vals) handle_error();

    int rows = 0, cols = -1;

    while (getline(&line, &cap, fp) != -1) {
        /* trim line: skip empties */
        char *p = line;
        while (*p && (*p==' ' || *p=='\t' || *p=='\r' || *p=='\n')) ++p;
        if (!*p) continue;

        /* count tokens in this row and push doubles */
        int this_cols = 0;
        for (char *tok = strtok(line, delims); tok; tok = strtok(NULL, delims)) {
            if (vals_len == vals_cap) {
                vals_cap *= 2;
                double *tmp = (double*)realloc(vals, vals_cap * sizeof(double));
                if (!tmp) { free(vals); handle_error(); }
                vals = tmp;
            }
            char *endptr = NULL;
            double v = strtod(tok, &endptr);
            if (endptr == tok) { free(vals); handle_error(); } /* not a number */
            vals[vals_len++] = v;
            ++this_cols;
        }
        if (this_cols == 0) continue;
        if (cols == -1) cols = this_cols;
        else if (this_cols != cols) { free(vals); handle_error(); } /* jagged */
        ++rows;
    }

    free(line);
    fclose(fp);

    if (rows <= 0 || cols <= 0) { free(vals); handle_error(); }
    if ((size_t)rows * (size_t)cols != vals_len) { free(vals); handle_error(); }

    Matrix X = create_matrix(rows, cols);
    memcpy(X.data, vals, vals_len * sizeof(double));
    free(vals);
    return X;
}

/*math*/
void similarity_matrix(const Matrix *X, Matrix *A) {
    /* A_ij = exp(-||xi - xj||^2 / 2) for i!=j; A_ii=0; symmetric */
    int n = X->rows;
    int d = X->cols;
    for (int i = 0; i < n; ++i) {
        mset(A, i, i, 0.0);
        for (int j = i + 1; j < n; ++j) {
            double dist2 = 0.0;
            for (int k = 0; k < d; ++k) {
                double diff = mget(X,i,k) - mget(X,j,k);
                dist2 += diff * diff;
            }
            double val = exp(-0.5 * dist2);
            mset(A, i, j, val);
            mset(A, j, i, val);
        }
    }
}

/* D is diagonal: D_ii = sum_j A_ij */
void degree_matrix(const Matrix *A, Matrix *D) {
    int n = A->rows;
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j) {
            s += mget(A, i, j);
        }
        mset(D, i, i, s);  // D is zero-initialized, so no need to zero the row
    }
}

void normalized_sim_matrix(const Matrix *A, const Matrix *D, Matrix *W) {
    /* W = D^{-1/2} A D^{-1/2} : W_ij = A_ij / (sqrt(d_i)*sqrt(d_j)) */
    int n = A->rows;
    double *degree_inv_sqrt = (double*)malloc((size_t)n * sizeof(double));
    if (!degree_inv_sqrt) handle_error();

    for (int i = 0; i < n; ++i) {
        double di = mget(D, i, i);
        degree_inv_sqrt[i] = (di > 0.0) ? (1.0 / sqrt(di)) : 0.0;
    }

    for (int i = 0; i < n; ++i) {
        double si = degree_inv_sqrt[i];
        for (int j = 0; j < n; ++j) {
            double sj = degree_inv_sqrt[j];
            mset(W, i, j, mget(A, i, j) * si * sj);
        }
    }
    free(degree_inv_sqrt);
}

/*updating H*/

/* One SymNMF update step (β-damped):
 * H_next_ij = H_ij * ( (1-β) + β * ((W H)_ij / (H (H^T H))_ij + den_eps) )
 * We compute H(H^T H) (same math as (H H^T)H, but faster when k << n).
 */
void symnmf_update_once(const Matrix *W, const Matrix *H, Matrix *Hnext,
                        double beta, double den_eps)
{
    int n = H->rows, k = H->cols;

    Matrix WH    = create_matrix(n, k);
    Matrix HT    = create_matrix(k, n);
    Matrix HtH   = create_matrix(k, k);
    Matrix H_HtH = create_matrix(n, k);

    /* Numerator: W H */
    mult_matrix(W, H, &WH);

    /* Denominator: H (H^T H) */
    transpose_matrix(H, &HT);
    mult_matrix(&HT, H, &HtH);
    mult_matrix(H, &HtH, &H_HtH);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            double denom  = mget(&H_HtH,i,j) + den_eps;
            double weight = (1.0 - beta) + beta * ( mget(&WH,i,j) / denom );
            double val    = mget(H,i,j) * weight;
            if (val < 0.0) val = 0.0;            /* keep nonnegative */
            mset(Hnext, i, j, val);
        }
    }

    free_matrix(&WH); free_matrix(&HT); free_matrix(&HtH); free_matrix(&H_HtH);
}

void symnmf_optimize(const Matrix *W, Matrix *H,
                     double eps, int max_iter, double beta, double den_eps)
{
    /* minimal shape sanity: W is n×n, H is n×k */
    if (W->rows != W->cols || W->rows != H->rows) {
        handle_error();
    }

    Matrix Hnext = create_matrix(H->rows, H->cols);

    for (int t = 0; t < max_iter; ++t) {
        /* one multiplicative update: H -> Hnext */
        symnmf_update_once(W, H, &Hnext, beta, den_eps);

        /* check convergence */
        double diff = frobenius_diff_sq(&Hnext, H);
        if (diff < eps) {
            /* copy latest into H, then stop */
            size_t nbytes = (size_t)H->rows * (size_t)H->cols * sizeof(double);
            memcpy(H->data, Hnext.data, nbytes);
            break;
        }

        /* swap buffers: next becomes current, reuse the other as scratch */
        Matrix tmp = *H;
        *H = Hnext;
        Hnext = tmp;
    }

    free_matrix(&Hnext);
}

/*print matrices*/

static void print_matrix(const Matrix *M) {
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            if (j) putchar(',');
            printf("%.4f", mget(M,i,j));
        }
        putchar('\n');
    }
}

/*main*/

int main(int argc, char **argv) {

    if (argc != 3) {
        handle_error();                    
    }

    const char *goal = argv[1];
    const char *file = argv[2];

    /*Validate goal*/
    if (!(strcmp(goal, "sym") == 0 ||
          strcmp(goal, "ddg") == 0 ||
          strcmp(goal, "norm") == 0)) {
        handle_error();
    }

    /*Load and validate data*/
    Matrix X = load_points(file);  /* n×d */
    if (X.data == NULL || X.rows <= 0 || X.cols <= 0) {
        if (X.data != NULL) free_matrix(&X);   //in case of allocating partial memory: free allocation
        handle_error();
    }

    /* Implementation */
    if (strcmp(goal, "sym") == 0) {
        Matrix A = create_matrix(X.rows, X.rows);
        similarity_matrix(&X, &A);
        print_matrix(&A);
        free_matrix(&A);

    } else if (strcmp(goal, "ddg") == 0) {
        Matrix A = create_matrix(X.rows, X.rows);
        Matrix D = create_matrix(X.rows, X.rows);
        similarity_matrix(&X, &A);
        degree_matrix(&A, &D);
        print_matrix(&D);
        free_matrix(&D);
        free_matrix(&A);

    } else { /* norm */
        Matrix A = create_matrix(X.rows, X.rows);
        Matrix D = create_matrix(X.rows, X.rows);
        Matrix W = create_matrix(X.rows, X.rows);
        similarity_matrix(&X, &A);
        degree_matrix(&A, &D);
        normalized_sim_matrix(&A, &D, &W);
        print_matrix(&W);
        free_matrix(&W);
        free_matrix(&D);
        free_matrix(&A);

    }

    free_matrix(&X);
    return 0;
}



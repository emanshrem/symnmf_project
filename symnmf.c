#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "symnmf.h"

/* We store matrices as one flat, contiguous double[] in row-major order.
 * This needs only one allocation/free, improves CPU cache locality, and
 * makes bulk ops faster */

/* error handling */
static void handle_error(void) {
    printf("An Error Has Occurred");
    exit(1);
}

/* matrix helpers */
Matrix create_matrix(int r, int c) {
     /*Allocate an r×c zero-initialized matrix as one contiguous row-major block*/
    Matrix M;
    size_t n;
    M.rows = r;
    M.cols = c;
    n = (size_t)r * (size_t)c;
    M.data = (double*)calloc(n, sizeof(double));
    if (!M.data) handle_error();
    return M;
}

void free_matrix(Matrix *M) {
    /*Release the matrix's contiguous data*/
    if (M && M->data) {
        free(M->data);
        M->data = NULL;
    }
    M->rows = M->cols = 0;
}

double mget(const Matrix *M, int i, int j) {
    /*Read element (i,j) from a row-major matrix*/
    return M->data[(size_t)i * (size_t)M->cols + (size_t)j];
}

void mset(Matrix *M, int i, int j, double v) {
    /*Write value v into element (i,j) of a row-major matrix*/
    M->data[(size_t)i * (size_t)M->cols + (size_t)j] = v;
}

void mult_matrix(const Matrix *A, const Matrix *B, Matrix *C) {
    /* * Compute C = A × B for row-major matrices.
    * Preconditions:
    *   - A is (r × m), B is (m × c), C is preallocated (r × c)
    *   - If shapes don’t match, handle_error() aborts.*/
    int r = A->rows;
    int m = A->cols;
    int c = B->cols;
    int i, j, k;
    double s;

    /* sizes must match, and output shape must be right */
    if (B->rows != m || C->rows != r || C->cols != c) {
        handle_error();
    }

    for (i = 0; i < r; ++i) {
        for (k = 0; k < c; ++k) {
            s = 0.0;
            for (j = 0; j < m; ++j) {
                s += mget(A, i, j) * mget(B, j, k);
            }
            mset(C, i, k, s);
        }
    }
}

void transpose_matrix(const Matrix *A, Matrix *AT) {
    /* * Compute AT = A^T for row-major matrices.
     * Preconditions:
     *   - A is (r x c); AT is preallocated as (c x r).
     *   - If shapes don’t match, handle_error() aborts.*/
    int r = A->rows;
    int c = A->cols;
    int i, j;

    if (AT->rows != c || AT->cols != r) {
        handle_error();
    }

    for (i = 0; i < r; ++i) {
        for (j = 0; j < c; ++j) {
            mset(AT, j, i, mget(A, i, j));
        }
    }
}


double frobenius_diff_sq(const Matrix *A, const Matrix *B) {
    /* Squared Frobenius norm: ||A - B||_F^2 */
    int r = A->rows, c = A->cols;
    int i, j;
    double s = 0.0, d;

    for (i = 0; i < r; ++i) {
        for (j = 0; j < c; ++j) {
            d = mget(A,i,j) - mget(B,i,j);
            s += d * d;
        }
    }
    return s;
}


static double *read_vals_rows_cols(const char *path, int *rows, int *cols, size_t *vals_len_out) {
    /*helper: read file -> flat values + rows/cols*/
    FILE *fp; const char *delims = " ,\t\r\n"; char buffer[8192];
    size_t vals_cap = 1024, vals_len = 0; double *vals, *tmp, v;
    char *p, *tok, *endptr; int this_cols;

    fp = fopen(path, "r"); if (!fp) handle_error();
    vals = (double*)malloc(vals_cap * sizeof(double)); if (!vals) handle_error();
    *rows = 0; *cols = -1;

    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        p = buffer; while (*p==' '||*p=='\t'||*p=='\r'||*p=='\n') ++p; if (!*p) continue;
        this_cols = 0; tok = strtok(buffer, delims);
        while (tok) {
            if (vals_len == vals_cap) { vals_cap *= 2u; tmp = (double*)realloc(vals, vals_cap * sizeof(double)); if (!tmp) { free(vals); handle_error(); } vals = tmp; }
            endptr = NULL; v = strtod(tok, &endptr); if (endptr == tok) { free(vals); handle_error(); }
            vals[vals_len++] = v; ++this_cols; tok = strtok(NULL, delims);
        }
        if (this_cols == 0) continue;
        if (*cols == -1) *cols = this_cols; else if (this_cols != *cols) { free(vals); handle_error(); }
        ++(*rows);
    }

    fclose(fp);
    if (*rows <= 0 || *cols <= 0) { free(vals); handle_error(); }
    if ((size_t)(*rows) * (size_t)(*cols) != vals_len) { free(vals); handle_error(); }
    *vals_len_out = vals_len; return vals;
}

Matrix load_points(const char *path) {
    /*public: build Matrix from file contents*/
    int rows, cols; size_t vals_len; double *vals; Matrix X;
    vals = read_vals_rows_cols(path, &rows, &cols, &vals_len);
    X = create_matrix(rows, cols); memcpy(X.data, vals, vals_len * sizeof(double));
    free(vals); return X;
}


/* math */
void similarity_matrix(const Matrix *X, Matrix *A) {
    /* A_ij = exp(-||xi - xj||^2 / 2) for i!=j; A_ii=0; symmetric */
    int n = X->rows;
    int d = X->cols;
    int i, j, k;
    double dist2, diff, val;

    for (i = 0; i < n; ++i) {
        mset(A, i, i, 0.0);
        for (j = i + 1; j < n; ++j) {
            dist2 = 0.0;
            for (k = 0; k < d; ++k) {
                diff = mget(X,i,k) - mget(X,j,k);
                dist2 += diff * diff;
            }
            val = exp(-0.5 * dist2);
            mset(A, i, j, val);
            mset(A, j, i, val);
        }
    }
}

void degree_matrix(const Matrix *A, Matrix *D) {
    int n = A->rows;
    int i, j;
    double s;

    for (i = 0; i < n; ++i) {
        s = 0.0;
        for (j = 0; j < n; ++j) {
            s += mget(A, i, j);
        }
        /* D is zero-initialized, so no need to zero the row */
        mset(D, i, i, s);
    }
}

void normalized_sim_matrix(const Matrix *A, const Matrix *D, Matrix *W) {
    /* W = D^{-1/2} A D^{-1/2} : W_ij = A_ij / (sqrt(d_i)*sqrt(d_j)) */
    int n = A->rows;
    double *degree_inv_sqrt;
    int i, j;
    double di, si, sj;

    degree_inv_sqrt = (double*)malloc((size_t)n * sizeof(double));
    if (!degree_inv_sqrt) handle_error();

    for (i = 0; i < n; ++i) {
        di = mget(D, i, i);
        degree_inv_sqrt[i] = (di > 0.0) ? (1.0 / sqrt(di)) : 0.0;
    }

    for (i = 0; i < n; ++i) {
        si = degree_inv_sqrt[i];
        for (j = 0; j < n; ++j) {
            sj = degree_inv_sqrt[j];
            mset(W, i, j, mget(A, i, j) * si * sj);
        }
    }
    free(degree_inv_sqrt);
}


void symnmf_update_once(const Matrix *W, const Matrix *H, Matrix *Hnext,
                        double beta, double den_eps)
{
    /* H_next_ij = H_ij * ((1-β) + β * ((W H)_ij / (H (H^T H))_ij + den_eps))
    * We compute H(H^T H) (same math as (H H^T)H, but faster when k << n).
    */
    int n = H->rows, k = H->cols;
    Matrix WH    = create_matrix(n, k);
    Matrix HT    = create_matrix(k, n);
    Matrix HtH   = create_matrix(k, k);
    Matrix H_HtH = create_matrix(n, k);
    int i, j;
    double denom, weight, val;

    /* Numerator: W H */
    mult_matrix(W, H, &WH);

    /* Denominator: H (H^T H) */
    transpose_matrix(H, &HT);
    mult_matrix(&HT, H, &HtH);
    mult_matrix(H, &HtH, &H_HtH);

    for (i = 0; i < n; ++i) {
        for (j = 0; j < k; ++j) {
            denom  = mget(&H_HtH,i,j) ;
            if (denom<=0.0) denom = den_eps;
            weight = (1.0 - beta) + beta * ( mget(&WH,i,j) / denom );
            val    = mget(H,i,j) * weight;
            if (val < 0.0) val = 0.0; /* keep nonnegative */
            mset(Hnext, i, j, val);
        }
    }

    free_matrix(&WH);
    free_matrix(&HT);
    free_matrix(&HtH);
    free_matrix(&H_HtH);
}

void symnmf_optimize(const Matrix *W, Matrix *H,
                     double eps, int max_iter, double beta, double den_eps)
{
    Matrix Hnext; 
    int t;
    double diff;
    size_t nbytes;

    /* minimal shape sanity: W is n×n, H is n×k */
    if (W->rows != W->cols || W->rows != H->rows) {
        handle_error();
    }

    Hnext = create_matrix(H->rows, H->cols);

    for (t = 0; t < max_iter; ++t) {
        /* one multiplicative update: H -> Hnext */
        symnmf_update_once(W, H, &Hnext, beta, den_eps);

        /* check convergence */
        diff = frobenius_diff_sq(&Hnext, H);
        if (diff < eps) {
            /* copy latest into H, then stop */
            nbytes = (size_t)H->rows * (size_t)H->cols * sizeof(double);
            memcpy(H->data, Hnext.data, nbytes);
            break;
        }

        /* swap buffers: next becomes current, reuse the other as scratch */
        {
            Matrix tmp = *H;
            *H = Hnext;
            Hnext = tmp;
        }
    }

    free_matrix(&Hnext);
}


static void print_matrix(const Matrix *M) {
    /* print matrices */
    int i, j;
    for (i = 0; i < M->rows; ++i) {
        for (j = 0; j < M->cols; ++j) {
            if (j) putchar(',');
            printf("%.4f", mget(M,i,j));
        }
        putchar('\n');
    }
}


/*goal runners*/
static void run_sym_from_X(Matrix *Xp) {
    Matrix X = *Xp;
    {
        Matrix A = create_matrix(X.rows, X.rows);
        similarity_matrix(&X, &A); print_matrix(&A); free_matrix(&A);
    }
}

static void run_ddg_from_X(Matrix *Xp) {
    Matrix X = *Xp;
    {
        Matrix A = create_matrix(X.rows, X.rows);
        Matrix D = create_matrix(X.rows, X.rows);
        similarity_matrix(&X, &A); degree_matrix(&A, &D);
        print_matrix(&D); free_matrix(&D); free_matrix(&A);
    }
}

static void run_norm_from_X(Matrix *Xp) {
    Matrix X = *Xp;
    {
        Matrix A = create_matrix(X.rows, X.rows);
        Matrix D = create_matrix(X.rows, X.rows);
        Matrix W = create_matrix(X.rows, X.rows);
        similarity_matrix(&X, &A); degree_matrix(&A, &D);
        normalized_sim_matrix(&A, &D, &W);
        print_matrix(&W); free_matrix(&W); free_matrix(&D); free_matrix(&A);
    }
}

/* main (only delegates to the runners) */
int main(int argc, char **argv) {
    if (argc != 3) handle_error();

    {
        const char *goal = argv[1], *file = argv[2];
        Matrix X;
        int g_sym, g_ddg, g_norm;

        g_sym  = (strcmp(goal, "sym")  == 0);
        g_ddg  = (strcmp(goal, "ddg")  == 0);
        g_norm = (strcmp(goal, "norm") == 0);
        if (!(g_sym || g_ddg || g_norm)) handle_error();

        X = load_points(file);
        if (X.data == NULL || X.rows <= 0 || X.cols <= 0) {
            /*If the loaded matrix is invalid, free any allocated memory, then abort with the standard error*/
            if (X.data) free_matrix(&X);
            handle_error();
        }

        if (g_sym)      run_sym_from_X(&X);
        else if (g_ddg) run_ddg_from_X(&X);
        else            run_norm_from_X(&X);

        free_matrix(&X);
    }
    return 0;
}





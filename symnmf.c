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
    if (M && M->data) {
        free(M->data);
        M->data = NULL;
    }
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

/* Squared Frobenius norm: ||A - B||_F^2 */
double frobenius_diff_sq(const Matrix *A, const Matrix *B) {
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

/* file loading: datapoints X
 * Accepts commas and/or spaces as separators
 */
Matrix load_points(const char *path) {
    FILE *fp = fopen(path, "r");
    const char *delims = " ,\t\r\n";
    char buffer[8192];
    size_t vals_cap = 1024, vals_len = 0;
    double *vals;
    int rows = 0, cols = -1;

    char *p, *tok;
    int this_cols;
    char *endptr;
    double v;

    if (!fp) handle_error();

    vals = (double*)malloc(vals_cap * sizeof(double));
    if (!vals) handle_error();

    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        /* trim-left whitespace; skip empty lines */
        p = buffer;
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
        if (!*p) continue;

        /* count tokens in this row and push doubles */
        this_cols = 0;
        tok = strtok(buffer, delims);
        while (tok) {
            if (vals_len == vals_cap) {
                double *tmp;
                vals_cap *= 2u;
                tmp = (double*)realloc(vals, vals_cap * sizeof(double));
                if (!tmp) { free(vals); handle_error(); }
                vals = tmp;
            }
            endptr = NULL;
            v = strtod(tok, &endptr);
            if (endptr == tok) { free(vals); handle_error(); } /* not a number */
            vals[vals_len++] = v;
            ++this_cols;
            tok = strtok(NULL, delims);
        }

        if (this_cols == 0) continue;
        if (cols == -1) cols = this_cols;
        else if (this_cols != cols) { free(vals); handle_error(); } /* jagged */
        ++rows;
    }

    fclose(fp);

    if (rows <= 0 || cols <= 0) { free(vals); handle_error(); }
    if ((size_t)rows * (size_t)cols != vals_len) { free(vals); handle_error(); }

    {
        Matrix X = create_matrix(rows, cols);
        memcpy(X.data, vals, vals_len * sizeof(double));
        free(vals);
        return X;
    }
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

/* D is diagonal: D_ii = sum_j A_ij */
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

/* H_next_ij = H_ij * ((1-β) + β * ((W H)_ij / (H (H^T H))_ij + den_eps))
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

/* print matrices */
static void print_matrix(const Matrix *M) {
    int i, j;
    for (i = 0; i < M->rows; ++i) {
        for (j = 0; j < M->cols; ++j) {
            if (j) putchar(',');
            printf("%.4f", mget(M,i,j));
        }
        putchar('\n');
    }
}

/* main */
int main(int argc, char **argv) {

    if (argc != 3) {
        handle_error();
    }

    {
        const char *goal = argv[1];
        const char *file = argv[2];
        Matrix X;

        /* Validate goal */
        if (!(strcmp(goal, "sym") == 0 ||
              strcmp(goal, "ddg") == 0 ||
              strcmp(goal, "norm") == 0)) {
            handle_error();
        }

        /* Load and validate data */
        X = load_points(file);  /* n×d */
        if (X.data == NULL || X.rows <= 0 || X.cols <= 0) {
            if (X.data != NULL) free_matrix(&X);
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
    }
    return 0;
}




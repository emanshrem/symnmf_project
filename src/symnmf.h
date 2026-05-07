#ifndef SYMNMF_H
#define SYMNMF_H

/* Matrix struct */
typedef struct {
    int rows, cols;
    double *data;
} Matrix;


/* Matrix helpers */
Matrix create_matrix(int r, int c);
void free_matrix(Matrix *M);
double mget(const Matrix *M, int i, int j);
void mset(Matrix *M, int i, int j, double v);
void mult_matrix(const Matrix *A, const Matrix *B, Matrix *C);
void transpose_matrix(const Matrix *A, Matrix *AT);
double frobenius_diff_sq(const Matrix *A, const Matrix *B);

/* File helpers */
double *read_vals_rows_cols(const char *path, int *rows, int *cols, size_t *vals_len_out);
Matrix load_points(const char *path);

/* Math functions */
void similarity_matrix(const Matrix *X, Matrix *A);
void degree_matrix(const Matrix *A, Matrix *D);
void normalized_sim_matrix(const Matrix *A, const Matrix *D, Matrix *W);

/* SymNMF optimization */
void symnmf_update_once(const Matrix *W, const Matrix *H, Matrix *Hnext,
                        double beta, double den_eps);
void symnmf_optimize(const Matrix *W, Matrix *H,
                     double eps, int max_iter, double beta, double den_eps);

/* Utility */
void print_matrix(const Matrix *M);

/* Goal runners */
void run_sym_from_X(Matrix *Xp);
void run_ddg_from_X(Matrix *Xp);
void run_norm_from_X(Matrix *Xp);

#endif /* SYMNMF_H */

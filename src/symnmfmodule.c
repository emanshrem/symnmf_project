#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"   /* Matrix, create/free, similarity_matrix, degree_matrix,
                         normalized_sim_matrix, symnmf_optimize, mget/mset, etc. */

static void handle_error(void) {
    PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred"); }

/* Convert a Python list-of-lists (NxD) of floats to Matrix (n x d). */
static int pylist_to_matrix(PyObject *obj, Matrix *out) {
    /* Acquire a fast wrapper for the outer sequence (list of rows). */
    PyObject *rows_fast = PySequence_Fast(obj, "expected a list of lists");
    if (!rows_fast) {
        /* A specific exception (TypeError) is already set by PySequence_Fast.
           handle_error() will only set a generic error if none is set. */
        handle_error();
        return 0;
    }

    /* Number of rows(n),reject empty input. */
    Py_ssize_t n = PySequence_Fast_GET_SIZE(rows_fast);
    if (n <= 0) {
        Py_DECREF(rows_fast);
        handle_error();
        return 0;
    }

    /* Determine number of columns (d) from the first row. */
    PyObject *row0 = PySequence_Fast_GET_ITEM(rows_fast, 0);  /* borrowed */
    PyObject *row_fast = PySequence_Fast(row0, "rows must be sequences");
    if (!row_fast) {
        Py_DECREF(rows_fast);
        handle_error();
        return 0;
    }

    Py_ssize_t d = PySequence_Fast_GET_SIZE(row_fast);
    Py_DECREF(row_fast);
    if (d <= 0) {
        Py_DECREF(rows_fast);
        handle_error();
        return 0;
    }

    /* Allocate the output Matrix (contiguous row-major n*d). */
    *out = create_matrix((int)n, (int)d);
    

    /* Iterate over rows. For each row, wrap it with PySequence_Fast so we can
       index quickly and validate its length equals d (rectangular check). */
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *row = PySequence_Fast_GET_ITEM(rows_fast, i);  /* borrowed */
        row_fast = PySequence_Fast(row, "rows must be sequences");
        if (!row_fast) {
            Py_DECREF(rows_fast);
            free_matrix(out);
            handle_error();
            return 0;
        }

        /* Every row must have exactly d items. */
        if (PySequence_Fast_GET_SIZE(row_fast) != d) {
            Py_DECREF(row_fast);
            Py_DECREF(rows_fast);
            free_matrix(out);
            handle_error();
            return 0;
        }

        /* Convert each value in the row to double and store into Matrix. */
        for (Py_ssize_t j = 0; j < d; ++j) {
            PyObject *v = PySequence_Fast_GET_ITEM(row_fast, j);  

            /* PyFloat_AsDouble accepts floats, ints, and objects with float.
               On failure it sets an exception */
            double x = PyFloat_AsDouble(v);
            if (PyErr_Occurred()) {
                Py_DECREF(row_fast);
                Py_DECREF(rows_fast);
                free_matrix(out);
                handle_error();
                return 0;
            }

            /*store: data[i*d + j] inside mset(). */
            mset(out, (int)i, (int)j, x);
        }

        /* Release the per-row fast wrapper before moving to the next row. */
        Py_DECREF(row_fast);
    }

    /* Release the outer fast wrapper and report success. */
    Py_DECREF(rows_fast);
    return 1;
}

/* Convert Matrix to Python list-of-lists. */
static PyObject* matrix_to_pylist(const Matrix *M) {
    PyObject *outer = PyList_New(M->rows);
    if (!outer) {
        return NULL; } 
    for (int i = 0; i < M->rows; ++i) {
        PyObject *inner = PyList_New(M->cols);
        if (!inner) {
            Py_DECREF(outer);
            return NULL; }
        for (int j = 0; j < M->cols; ++j) {
            PyObject *num = PyFloat_FromDouble(mget(M, i, j));
            if (!num) { 
                Py_DECREF(inner);
                Py_DECREF(outer); return NULL; }
            PyList_SET_ITEM(inner, j, num); 
        }
        PyList_SET_ITEM(outer, i, inner);  
    }
    return outer;
}

/*Functions wrappers*/

/*sym(x) to A*/
static PyObject* py_sym(PyObject *self, PyObject *args) {
    PyObject *Xobj;
    if (!PyArg_ParseTuple(args, "O", &Xobj)) {
        return NULL; }
    Matrix X; 
    if (!pylist_to_matrix(Xobj, &X)) { 
        return NULL; }
    Matrix A = create_matrix(X.rows, X.rows);
    similarity_matrix(&X, &A);

    PyObject *out = matrix_to_pylist(&A);
    free_matrix(&A);
    free_matrix(&X);
    if (!out) {
        handle_error(); }
    return out;
}

/* ddg(X) to D*/
static PyObject* py_ddg(PyObject *self, PyObject *args) {
    PyObject *Xobj;
    if (!PyArg_ParseTuple(args, "O", &Xobj)) {
        return NULL;
    } 
    Matrix X; 
    if (!pylist_to_matrix(Xobj, &X)) {
        return NULL; }
    Matrix A = create_matrix(X.rows, X.rows);
    Matrix D = create_matrix(X.rows, X.rows);
    similarity_matrix(&X, &A);
    degree_matrix(&A, &D);
    PyObject *out = matrix_to_pylist(&D);
    free_matrix(&D); free_matrix(&A); free_matrix(&X);
    if (!out) {
        handle_error(); }
    return out;
}

/* norm(X) to W = D^{-1/2} A D^{-1/2} */
static PyObject* py_norm(PyObject *self, PyObject *args) {
    PyObject *Xobj;
    if (!PyArg_ParseTuple(args, "O", &Xobj)) return NULL;

    Matrix X; if (!pylist_to_matrix(Xobj, &X)) return NULL;
    Matrix A = create_matrix(X.rows, X.rows);
    Matrix D = create_matrix(X.rows, X.rows);
    Matrix W = create_matrix(X.rows, X.rows);

    similarity_matrix(&X, &A);
    degree_matrix(&A, &D);
    normalized_sim_matrix(&A, &D, &W);

    PyObject *out = matrix_to_pylist(&W);
    free_matrix(&W); free_matrix(&D); free_matrix(&A); free_matrix(&X);
    if (!out) handle_error();
    return out;
}

/*symnmf(H0, W, eps, max_iter, beta, den_eps) to H final */
static PyObject* py_symnmf(PyObject *self, PyObject *args) {
    PyObject *H0obj, *Wobj;
    double eps, beta, den_eps;
    int max_iter;

    if (!PyArg_ParseTuple(args, "OOdidd", &H0obj, &Wobj, &eps, &max_iter, &beta, &den_eps)) {
        return NULL; }

    Matrix H, W;
    if (!pylist_to_matrix(H0obj, &H)) {
        return NULL; }
    if (!pylist_to_matrix(Wobj,  &W)) { 
        free_matrix(&H);
        return NULL; }

    /* optimize in-place over H using W */
    symnmf_optimize(&W, &H, eps, max_iter, beta, den_eps);

    PyObject *out = matrix_to_pylist(&H);
    free_matrix(&H); free_matrix(&W);
    if (!out) {
        handle_error(); }
    return out;
}

/*Module table & init*/
/*METH_VARARGS — calling convention where CPython passes all positional
args as a single tuple (PyObject *args), to be unpacked with PyArg_ParseTuple.*/
static PyMethodDef SymNMFMethods[] = {
    {"sym",py_sym,METH_VARARGS,"Returns similarity matrix A from datapoints X."},
    {"ddg",py_ddg,METH_VARARGS,"Returns diagonal degree matrix D from datapoints X."},
    {"norm",py_norm,METH_VARARGS,"Returns normalized similarity matrix W from datapoints X."},
    {"symnmf",py_symnmf,METH_VARARGS,"Runs SymNMF from initial H0 and W; return final H."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmf",      /*module name as seen by Python */
    NULL,          /*doc string */
    -1,            /*per-interpreter state size */
    SymNMFMethods
};

PyMODINIT_FUNC PyInit_symnmf(void) {
    return PyModule_Create(&symnmfmodule);
}

#include <Python.h>
#include <stdbool.h>
#include <numpy/arrayobject.h>
//#include <numpy/ndarrayobject.h>

#include "longdouble.h"

static PyObject *LongDoubleError;

static PyObject *Py_string2longdouble(PyObject *self, PyObject *args) {
    PyObject *string_obj = NULL;
    char *_string;
    long double *_longdouble;
    PyObject *longdouble_obj = NULL;
    npy_intp dims[1];
    
    if (!PyArg_ParseTuple(args,
			  (char*)"O",
			  &string_obj)) {
	fprintf(stderr,"Failed to parse string object.\n");
	return NULL;
    }

    // Raise an exception if this isn't a string...
    
    if (!PyString_Check(string_obj)) {
	PyErr_SetString(LongDoubleError, "Input to string2longdouble must be a string.");
	return NULL;
    }

    // extract string object into c string

    _string = PyString_AsString(string_obj);

    // make the output object -- 0 length array.  
    dims[0] = 1;
    longdouble_obj = PyArray_ZEROS(0,dims,NPY_LONGDOUBLE,0);

    // get a pointer to the data...
    _longdouble = (long double *) PyArray_DATA(longdouble_obj);

    // Do the conversion
    
    if (string2longdouble(_string, _longdouble) < 0) {
	PyErr_SetString(LongDoubleError,"Error with conversion.");
	// do I need to decrement the ref counter??
	return NULL;
    }
    
    return PyArray_Return((PyArrayObject *)longdouble_obj);
}

static PyObject *Py_longdouble2string(PyObject *self, PyObject *args) {
    PyObject *longdouble_obj = NULL;
    long double _longdouble;
    char _string[STRINGLEN];
    long ndigit = 0;

    if (!PyArg_ParseTuple(args,
			  (char*)"Ol",
			  &longdouble_obj,
			  &ndigit)) {
	fprintf(stderr,"Failed to parse longdouble object.\n");
	return NULL;
    }

    // raise an exception if this isn't a scalar...
    if (!PyArray_CheckScalar(longdouble_obj)) {
	PyErr_SetString(LongDoubleError,"Input long double must be a scalar.");
	return NULL;
    }

    // cast to long double
    PyArray_CastScalarToCtype(longdouble_obj, &_longdouble, PyArray_DescrFromType(NPY_LONGDOUBLE));

    // do the conversion
    if (longdouble2string(_longdouble, _string, ndigit) < 0) {
	PyErr_SetString(LongDoubleError,"Error with conversion.");
	return NULL;
    }

    return PyString_FromString(_string);
}

static PyObject *Py_doubledouble2longdouble(PyObject *self, PyObject *args) {
    PyObject *doubledouble_obj = NULL;
    double _double[2];
    long double *_longdouble;
    PyObject *longdouble_obj = NULL;
    npy_intp dims[1];
    Py_ssize_t i;

    if (!PyArg_ParseTuple(args,
			  (char*)"O",
			  &doubledouble_obj)) {
	fprintf(stderr,"Failed to parse doubledouble tuple.\n");
	return NULL;
    }

    // Raise an exception if it isn't a tuple with 2 guys
    if (!PyTuple_Check(doubledouble_obj)) {
	PyErr_SetString(LongDoubleError,"Input to doubledouble2longdouble must be a tuple.");
	return NULL;
    }

    if (PyTuple_Size(doubledouble_obj) != 2) {
	PyErr_SetString(LongDoubleError,"Input to doubledouble2longdouble must be a tuple with 2 objects.");
	return NULL;
    }

    for (i=0;i<2;i++) {
	_double[i] = PyFloat_AsDouble(PyTuple_GetItem(doubledouble_obj,i));
    }

    // make the output object -- 0 length array
    dims[0] = 1;
    longdouble_obj = PyArray_ZEROS(0,dims,NPY_LONGDOUBLE,0);

    // get a pointer to the data...
    _longdouble = (long double *) PyArray_DATA(longdouble_obj);

    // Do the conversion
    if (doubledouble2longdouble(_double, _longdouble) < 0) {
		PyErr_SetString(LongDoubleError,"Error with conversion.");
	// do I need to decrement the ref counter??
	return NULL;
    }
    
    return PyArray_Return((PyArrayObject *)longdouble_obj);
}

static PyObject *Py_longdouble2doubledouble(PyObject *self, PyObject *args) {
    PyObject *longdouble_obj = NULL;
    long double _longdouble;
    double _double[2];
    PyObject *double_obj0;
    PyObject *double_obj1;
    PyObject *doubledouble_obj;

    if (!PyArg_ParseTuple(args,
			  (char*)"O",
			  &longdouble_obj)) {
	fprintf(stderr,"Failed to parse longdouble object.\n");
	return NULL;
    }

    // raise an exception if this isn't a scalar...
    if (!PyArray_CheckScalar(longdouble_obj)) {
	PyErr_SetString(LongDoubleError,"Input long double must be a scalar.");
	return NULL;
    }

    // cast to long double
    PyArray_CastScalarToCtype(longdouble_obj, &_longdouble, PyArray_DescrFromType(NPY_LONGDOUBLE));

    // do the conversion
    if (longdouble2doubledouble(_longdouble,_double) < 0) {
	PyErr_SetString(LongDoubleError,"Error with conversion.");
	return NULL;
    }

    // need to create a tuple here...

    // first need to create objects for the two doubles..
    double_obj0 = PyFloat_FromDouble(_double[0]);
    double_obj1 = PyFloat_FromDouble(_double[1]);

    doubledouble_obj = PyTuple_Pack(2, double_obj0, double_obj1);

    return doubledouble_obj;
}

static PyObject *Py_strings2longdoubles(PyObject *self, PyObject *args) {
    PyObject *strings_obj = NULL;
    PyObject *longdoubles_obj = NULL;
    PyObject *string_obj = NULL;
    long double *_longdoubles;
    char *_string;
    Py_ssize_t nstring;
    int i;
    npy_intp dims[1];

    if (!PyArg_ParseTuple(args,
			  (char*)"O",
			  &strings_obj)) {
	fprintf(stderr,"Failed to parse strings object.\n");
	return NULL;
    }

    // This object needs to be a list of strings...
    if (!PyList_Check(strings_obj)) {
	PyErr_SetString(LongDoubleError,"Input to strings2longdoubles must be a list.");
	return NULL;
    }

    // how many objects?
    nstring = PyList_Size(strings_obj);

    // make the output object...
    dims[0] = nstring;
    longdoubles_obj = PyArray_ZEROS(1,dims,NPY_LONGDOUBLE,0);

    _longdoubles = (long double *) PyArray_DATA(longdoubles_obj);

    // loop over input list
    for (i=0;i<nstring;i++) {
	string_obj = PyList_GetItem(strings_obj, i);
	if (!PyString_Check(string_obj)) {
	    PyErr_SetString(LongDoubleError,"Input list must all be strings.");
	    return NULL;
	}
	_string = PyString_AsString(string_obj);

	if (string2longdouble(_string, &_longdoubles[i]) < 0) {
	    PyErr_SetString(LongDoubleError,"Error with conversion");
	    return NULL;
	}
    }
    
    return PyArray_Return((PyArrayObject *)longdoubles_obj);
}


static PyMethodDef LongDouble_type_methods[] = {
    {"string2longdouble", (PyCFunction)Py_string2longdouble, METH_VARARGS, NULL},
    {"longdouble2string", (PyCFunction)Py_longdouble2string, METH_VARARGS, NULL},
    {"doubledouble2longdouble", (PyCFunction)Py_doubledouble2longdouble, METH_VARARGS, NULL},
    {"longdouble2doubledouble", (PyCFunction)Py_longdouble2doubledouble, METH_VARARGS, NULL},
    {"strings2longdoubles", (PyCFunction)Py_strings2longdoubles, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL} /* Sentinel */
};


#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
init_longdouble_pywrap(void)
{
    PyObject *m;

    m = Py_InitModule3("_longdouble_pywrap", LongDouble_type_methods,"Define longdouble type and methods.");
    if (m == NULL) {
	return;
    }

    LongDoubleError = PyErr_NewException("longdouble.error", NULL, NULL);
    Py_INCREF(LongDoubleError);
    PyModule_AddObject(m, "error", LongDoubleError);
	

    import_array();
}
    

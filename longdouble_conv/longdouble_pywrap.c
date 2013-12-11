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

    // cast to long double  -- need to make sure this isn't truncating
    PyArray_CastScalarToCtype(longdouble_obj, &_longdouble, PyArray_DescrFromType(NPY_LONGDOUBLE));

    // do the conversion
    if (longdouble2string(_longdouble, _string, ndigit) < 0) {
	PyErr_SetString(LongDoubleError,"Error with conversion.");
	return NULL;
    }

    return PyString_FromString(_string);
}


static PyMethodDef LongDouble_type_methods[] = {
    {"string2longdouble", (PyCFunction)Py_string2longdouble, METH_VARARGS, NULL},
    {"longdouble2string", (PyCFunction)Py_longdouble2string, METH_VARARGS, NULL},
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
    

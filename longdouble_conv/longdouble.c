#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <limits.h>

#include "longdouble.h"

int string2longdouble(char *_string, long double *_longdouble) {

    // just use sscanf.  Simple!
    
    if (sscanf(_string,"%Lf",_longdouble) != 1) {
	*_longdouble = -1;
	return -1;
    }

    return 0;
}

int longdouble2string(long double _longdouble, char *_string, long ndigit) {

    // this is not simple, sadly.
    char format[STRINGLEN];
    long exponent;
    
    int class = __fpclassifyl(_longdouble);

    if (class == FP_ZERO) {
	snprintf(_string,STRINGLEN,"0.0");
    } else if (class == FP_INFINITE) {
	if (isinfl(_longdouble) == 1) {
	    snprintf(_string,STRINGLEN,"+inf");
	} else snprintf(_string,STRINGLEN,"-inf");
    } else if (class == FP_NAN) {
	snprintf(_string,STRINGLEN,"nan");
    } else {
	// we have a normal fp number

	// record the sign and make positive
	/*if (_longdouble < 0.0) {
	    sign = '-';
	    _longdouble*=-1;
	}
	*/
	// find the exponent...
	exponent = (long) floorl(log10l(fabsl(_longdouble)));

	// if the exponent is < 5, it's shorter to do a straight print
	if (exponent >=0 && exponent < 5) {
	    // positive exponent... straight on
	    snprintf(format,STRINGLEN,"%%.%0ldLf", ndigit);
	} else if (exponent > -5) {
	    // negative ... expand
	    snprintf(format,STRINGLEN,"%%.%0ldLf", ndigit - exponent);
	} else {
	    // exponent version
	    snprintf(format,STRINGLEN,"%%.%0ldLe", ndigit);
	}
	snprintf(_string,STRINGLEN,format,_longdouble);

    }
    
    return 0;
}

int longdouble2doubledouble(long double _longdouble, double *_doubledouble) {
    _doubledouble[0] = (double)_longdouble;
    _doubledouble[1] = (double)(_longdouble - (long double) _doubledouble[0]);

    return 0;
}

int doubledouble2longdouble(double *_doubledouble, long double *_longdouble) {
    *_longdouble = (long double) _doubledouble[0] + (long double) _doubledouble[1];

    return 0;
}

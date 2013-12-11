#ifndef _LONGDOUBLE
#define _LONGDOUBLE

#define STRINGLEN   1000

int string2longdouble(char *_string, long double *_longdouble);
int longdouble2string(long double _longdouble, char *_string, long ndigit);
int longdouble2doubledouble(long double _longdouble, double *_doubledouble);
int doubledouble2longdouble(double *_doubledouble, long double *_longdouble);


#endif

"""
longdouble_conv

Copyright (C) 2013  Eli Rykoff, SLAC.  erykoff at gmail dot com

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""

import numpy as np
from . import _longdouble_pywrap

def string2longdouble(_string):
    return _longdouble_pywrap.string2longdouble(_string)

def longdouble2string(_longdouble,n=19):
    return _longdouble_pywrap.longdouble2string(_longdouble,n)

def doubledouble2longdouble(_doubledouble):
    return _longdouble_pywrap.doubledouble2longdouble(_doubledouble)

def longdouble2doubledouble(_longdouble):
    return _longdouble_pywrap.longdouble2doubledouble(_longdouble)

def string2doubledouble(_string):
    _longdouble = string2longdouble(_string)
    return longdouble2doubledouble(_longdouble)

def doubledouble2string(_doubledouble):
    _longdouble = doubledouble2longdouble(_doubledouble)
    return longdouble2string(_longdouble)

def strings2longdoubles(_strings,array=True):
    # I had intended to have this in the c code but this is much simpler, if a
    # bit slower in those cases where you need a list rather than a numpy
    # array.
    
    arr = _longdouble_pywrap.strings2longdoubles(_strings)

    if (not array):
        return arr.tolist()
    else :
        return arr

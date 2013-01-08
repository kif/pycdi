#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://forge.epn-campus.eu/projects/azimuthal
#
#    File: "$Id$"
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#


import cython
cimport numpy
import numpy
from cython.parallel cimport prange
from libc.math cimport sin, cos, atan2, sqrt, M_PI

cdef float cabs(numpy.complex64_t z) nogil:
    return sqrt(z.real**2 + z.imag**2) 
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def constrains_fourier( fourier not None,
                        intensity not None,
                       ):
    """
    Calculate the constrains in Fourier space 
    """
#    
    cdef numpy.complex64_t[:] cfourier = fourier.ravel()
    cdef float[:] cintensity = intensity.ravel() 
    cdef int i, size 
    cdef numpy.complex64_t data
    cdef float one_int, data_abs
    size  = fourier.size
    assert intensity.size == size
    for i in prange(size, nogil=True, schedule="static"):
                data = cfourier[i]  
                one_int = cintensity[i] 
                data_abs = cabs(data)    
                if (one_int > 0.0) and (data_abs!=0.0):
                    cfourier[i] = one_int*data/data_abs
    return fourier

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def constrains_real( vol not None,
                     last not None,
                     mask not None,
                     float scale=0.2  ):
    """
    Calculate the constrains in real space 
    """
    cdef numpy.complex64_t[:] cvol = vol.ravel()
    cdef numpy.complex64_t[:] clast = last.ravel()
    cdef numpy.int8_t[:] cmask = mask.ravel() 
    cdef int i, size 
    cdef numpy.complex64_t data
    size  = vol.size
    assert last.size == size
    assert mask.size == size
    for i in prange(size, nogil=True, schedule="static"):
                data = cvol[i]  
                if not (cmask[i]) or data.real<0:
                    cvol[i] = clast[i]-scale*data
    return vol

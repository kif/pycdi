#!/usr/bin/env python
#reconstruction code for HIO

import EdfFile
import numpy
import numpy.fft
import sys
from time import time, sleep
from os.path import isfile
from scipy.ndimage import gaussian_filter
import multiprocessing

import pycuda
import pycuda.autoinit
from pycuda.elementwise import ElementwiseKernel, VectorArg, ScalarArg
import pycuda.driver
import pycuda.gpuarray as gpuarray
import scikits.cuda.fft as cu_fft

def loadedf(filename, imgn=0):
    if isfile(filename):
        f = EdfFile.EdfFile(filename)
        return f.GetData(imgn)
    else:
        print "file ", filename, " does not exist!"
        return 0

def saveedf(filename, data, imgn=0):
    try:
        newf = EdfFile.EdfFile(filename)
        newf.WriteImage({}, data, imgn)
        print "file is saved to ", filename
        return
    except:
        print "file is not saved!"
        return
def magf(x):
    return abs(numpy.fft.fftn(x))

print "pycdi.py phase retrival program"
print "usage ./pycdi.py data file number of iterations (optional)"
print "example ./pycdi.py image.edf 2000"

try:
  filename = sys.argv[1]
except:
  print "ERROR!!! provide the data file"
  sys.exit()
try:
  N = int(sys.argv[2])
except:
  N = 10

sm = 512
threads = multiprocessing.cpu_count()
ss = 128
try:
  img = numpy.sqrt(numpy.load(filename))
except:
  print " ERROR!! problem loading the file " + filename
  sys.exit()

try:
  maski = loadedf(sys.argv[3])
  img[maski > 0] = 0
  print "using image mask"
except: pass
img = numpy.sqrt(numpy.load(filename))
n, m, k = shape = img.shape
size = img.size

#img=shift(img,(0.5,0.5,0),order=1,prefilter=False)
sq = 40
x, y, z = numpy.mgrid[-n / 2:n / 2, -m / 2:m / 2, -k / 2:k / 2]
r = numpy.sqrt(x ** 2 + y ** 2, z ** 2)
sx = 3#6
sy = 3#6
sz = 3#6
img[n / 2 - sx:n / 2 + sx, m / 2 - sy:m / 2 + sy, k / 2 - sz:k / 2 + sz] = 0
#img[r<6]=0
print n, m, k
print " Treating " + filename
print " Number of iterations " + str(N)
#HIO
beta = 0.9
gamma_1 = -1.0#not used
gamma_2 = 1.0 / beta


sobject = numpy.random.random_sample((n, m, k)).astype(numpy.float32)
#sobject=load('random.npy')
try:
  mask = numpy.zeros((n, m, k))
  #tmp=load('p1_3Dn2cp12_AreconstructionAverage.npy')
  #tmp=load('p1_3Dn2cp12_AreconstructionHIOfs2734.npy')
  gaussian_filter(tmp, 1, output=tmp)
  mask[tmp > 0.0001] = 1
  del tmp
  print "using an average reconstruction as a mask"
except:
  mask = numpy.zeros((n, m, k))
  x, y, z = numpy.mgrid[-n / 2:n / 2, -m / 2:m / 2, -k / 2:k / 2]
  r = numpy.sqrt(x ** 2 + y ** 2 + z ** 2)
  offset = 10
  #mask[r<=sq]=1#circular support
  mask[n / 2 - sq:n / 2 + sq, m / 2 - sq:m / 2 + sq, k / 2 - sq / 2:k / 2 + sq / 2] = 1# square support
  #mask[n/2-sq:n/2+sq,n/2-sq-offset:n/2+sq-offset]=1# square support
  print "using a square as a mask"

s_index = numpy.where(mask == 0)
sobject[s_index] = 0#put zeros outside the support 

todel = numpy.fft.fftshift(magf(sobject))
img = img / img.sum() * todel.sum()


del todel
del mask
del r, x, y, z

mag = numpy.fft.fftshift(img).astype(numpy.float32)
del img
#index = numpy.where(mag <= 0)#pixels where the data equal 0
indexp = numpy.where(mag > 0)#pixels where the data >0
#isobm=zeros((n,m,k),dtype=int8)
#isobm[index]=1
sobm = numpy.zeros((n, m, k), dtype=numpy.int8)
sobm[s_index] = 1
sobject = sobject.astype(numpy.complex64)
ctx = pycuda.autoinit.context
gpu_data = gpuarray.zeros(shape, numpy.complex64)
gpu_last = gpuarray.zeros(shape, numpy.complex64)
gpu_intensity = gpuarray.zeros(shape, numpy.float32)
gpu_mask = gpuarray.zeros(shape, numpy.int8)
plan = cu_fft.Plan(shape, numpy.complex64, numpy.complex64)

constrains_fourier = ElementwiseKernel([VectorArg(numpy.complex64, "fourier"), VectorArg(numpy.float32, "intensity") ],
        """
        float one_int, data_abs;
        pycuda::complex<float> data;
        
        data = fourier[i];
        one_int = intensity[i]; 
        data_abs = abs(data);    
        if ((one_int > 0.0) && (data_abs!=0.0))
            fourier[i] = one_int *data / data_abs;
        """, name="kfourier")
constrains_real = ElementwiseKernel([VectorArg(numpy.complex64, "vol"),
                                     VectorArg(numpy.complex64, "last"),
                                     VectorArg(numpy.int8, "mask"),
                                     ScalarArg(numpy.float32, "scale")],
        """
        pycuda::complex<float> data;
        
        data = vol[i] / (float) n;  
        if ((mask[i]>0) || (data.real()<0))
                    vol[i] = last[i]-scale*data;
        else
                    vol[i] = data;        
        """, name="kdirect")

#real_space = numpy.empty(sobject.shape, dtype=numpy.complex64)
#fourier_space = numpy.empty(sobject.shape, dtype=numpy.complex64)
#fft = fftw3f.Plan(real_space, fourier_space, direction='forward', nthreads=threads)
#ifft = fftw3f.Plan(fourier_space, real_space, direction='backward', nthreads=threads)
erra = []
errtmp = 0
nerr = 0
serr = 1e+7
result = numpy.zeros((n, m, k), dtype=numpy.complex64)
time0 = time()
ii = 0
tmpimg = numpy.zeros((n, m, k), dtype=numpy.float32)

ln = sq + 5
mags = mag[indexp].sum()
del indexp
s = 3
N2 = int(N * 0.7)
N3 = int(N * 0.7)

gpu_data.set(sobject.astype(numpy.complex64))
pycuda.driver.memcpy_dtod(gpu_last.gpudata, gpu_data.gpudata, gpu_data.nbytes)
gpu_intensity.set(mag)
gpu_mask.set(sobm)
#print real_space.nbytes
for i in range(N):
    t0 = time()
    cu_fft.fft(gpu_data, gpu_data, plan)
    constrains_fourier(gpu_data, gpu_intensity)
    cu_fft.ifft(gpu_data, gpu_data, plan, True)
    constrains_real(gpu_data, gpu_last, gpu_mask, beta)
    pycuda.driver.memcpy_dtod(gpu_last.gpudata, gpu_data.gpudata, gpu_data.nbytes)
    t1 = time()
    ctx.synchronize()
    t2 = time()
    print("With CUDA, the full loop took %.3fs but after sync %.3fs" % (t1 - t0, t2 - t0))

del tmpimg
print "it took", time() - time0, N / (time() - time0)
print "smallest error", serr, "number", nerr

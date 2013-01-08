#!/usr/bin/env python
#reconstruction code for HIO

import numpy
import numpy.fft
import EdfFile
import sys
from time import time, sleep
from os.path import isfile
from scipy.ndimage import gaussian_filter
import fftw3f
from CDI3D import isob, sob, errtd
import constrains
import multiprocessing

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

n, m, k = img.shape
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

s_index = where(mask == 0)
sobject[s_index] = 0#put zeros outside the support 

todel = numpy.fft.fftshift(magf(sobject))
img = img / img.sum() * todel.sum()


del todel
del mask
del r, x, y, z

mag = numpy.fft.fftshift(img).astype(float32)
del img
index = where(mag <= 0)#pixels where the data equal 0
indexp = where(mag > 0)#pixels where the data >0
#isobm=zeros((n,m,k),dtype=int8)
#isobm[index]=1
sobm = numpy.zeros((n, m, k), dtype=int8)
sobm[s_index] = 1

sobject = array(sobject, dtype=numpy.complex64)
real_space = numpy.empty(sobject.shape, dtype=numpy.complex64)
fourier_space = numpy.empty(sobject.shape, dtype=numpy.complex64)
fft = fftw3f.Plan(real_space, fourier_space, direction='forward', nthreads=threads)
ifft = fftw3f.Plan(fourier_space, real_space, direction='backward', nthreads=threads)

erra = []
errtmp = 0
nerr = 0
serr = 1e+7
result = numpy.zeros((n, m, k), dtype=numpy.complex64)
time0 = time()
ii = 0
tmpimg = numpy.zeros((n, m, k), dtype=float32)

#ln=copy(sq)#size of the array for visualisation
ln = sq + 5
mags = mag[indexp].sum()
del indexp
s = 3
#nm = 1.0 / (n * m * k)
N2 = int(N * 0.7)
N3 = int(N * 0.7)

real_space[:] = sobject.astype(numpy.complex64)
last = real_space.copy()
for i in range(N):
    fft()
    t0 = time()
#    isobject += isob(isobject, mag, sout, n, m, k)
#    print abs(fourier_space[115:118, 115:118, 115:118])
    constrains.constrains_fourier(fourier_space, mag)
    print("constrains in Fourier space took %.3fs" % (time() - t0))
#    print abs(fourier_space[115:118, 115:118, 115:118])
#    sys.exit()
    ifft()
#    isout *= nm
#    sobject += sob(sobject, isout, beta, sobm, n, m, k)
    t1 = time()
    constrains.constrains_real(real_space, last, sobm, beta)
    last = real_space.copy()
    t2 = time()
    print("constrains in Real space took %.3fs, the full loop took %.3fs" % (t2 - t1, t2 - t0))
#    errtmp = errtd(mag, sout, mags, n, m, k)
#    tmpimg += isout
#    if errtmp < serr:
#        nerr = copy(i)
#        ii = copy(i)
#        result -= result
#        result += isout
#        serr = copy(errtmp)
#    erra.append(errtmp)
#    print i, "%.4f" % erra[-1], "%.4f" % serr, ii, "%.4f" % s
#    if i < N2:
#        if (i + 1) / 20 == (i + 1) / 20.0:
#          s = 0.25 + 2.25 * exp(-(i + 1.0) / N3)
#          tmpimg = gaussian_filter(tmpimg, s)
#          sobm *= 0 #this is good
#          sobm[where(tmpimg < tmpimg.max() * 0.04)] += 1#this is good
#          tmpimg *= 0
#          errtmp = 1.0
#          serr = 2.0
#    else:
#        sobject[sobm > 0] = 0

#tmpimg/=N
del tmpimg
print "it took", time() - time0, N / (time() - time0)
print "smallest error", serr, "number", nerr
inputfile = filename.split(".")[0] + "_AreconstructionHIOfs" + str(int(serr * 10000)) + ".npy"
inputfilep = filename.split(".")[0] + "_AreconstructionHIOfsP" + str(int(serr * 10000)) + ".npy"

z = 0
while isfile(inputfile):
  inputfile = filename.split(".")[0] + "_AreconstructionHIOfs" + str(int(serr * 10000)) + str(z) + ".npy"
  inputfilep = filename.split(".")[0] + "_AreconstructionHIOfsP" + str(int(serr * 10000)) + str(z) + ".npy"
  z += 1

save(inputfile, abs(result))
save(inputfilep, arctan(imag(result) / real(result)))
savetxt(filename.split(".")[0] + "_error.dat", erra)


# coding: utf-8

# In[1]:

import pycuba
import numpy as np
import time
#load_ext fortranmagic


# In[22]:

def cuba(method,nD,nC,integrand,epsR,epsA):
    NDIM=nD;    NCOMP=nC;    EPSREL=epsR;    NNEW = 1000;    NMIN = 2;    FLATNESS = 50.;    KEY1 = 47;    KEY2 = 1;
    KEY3 = 1;    MAXPASS = 5;    BORDER = 0.;    MAXCHISQ = 10.;    MINDEVIATION = .25;    NGIVEN = 0;    LDXGIVEN = NDIM;
    NEXTRA = 0;    MINEVAL = 0;    MAXEVAL = 1000000;    KEY = 0;
    if (method == 'Vegas'):
        results=pycuba.Vegas(integrand,nD,verbose=2)
        return results['results'][0]['integral'],results['results'][0]['error'],results['results'][0]['prob']
    elif (method == 'Cuhre'):
        results=pycuba.Cuhre(integrand,nD,key=KEY,verbose=2)
    else:
        print('Please enter Vegas or Cuhre')
        return 0
    if (results['fail'] != 0):
        print('Failed to Converge')
        keys = ['nregions', 'neval', 'fail']
        keys = list(filter(results.has_key, keys))
        text = ["%s %d" % (k, results[k]) for k in keys]
        print("%s RESULT:\t" % method.upper() + "\t".join(text))
        for comp in results['results']:
            print("%s RESULT:\t" % method.upper() +                   "%(integral).8f +- %(error).8f\tp = %(prob).3f\n" % comp)
    else:
#        return results
        return results['results'][0]['integral'],results['results'][0]['error'],results['results'][0]['prob']


# In[23]:

def gauss_int(ndim, xx, ncomp, ff,userdata):
    x=np.array([xx[i] for i in range(ndim.contents.value)])
    a = np.array([0.,0.,0.])*10**4
    b = np.array([1.,1.,1.])*10**5
    x=a+(b-a)*x #scale the coordinates
    result=np.exp(-np.dot(x,x)/2.)/(np.sqrt(2.0*np.pi)**3)
    jacobian=np.prod(b-a)*8
    result*=jacobian
    ff[0] = result 
    return 0


# %%fortran
# subroutine ps22_tf_int(ndim, x, ncomp, f)
#       implicit none
#       integer       :: ndim, ncomp
#       real*8 :: x(ndim), f(ncomp)
#       real*8 :: k, q, pmin, pmax, y, f2, rcos, p
#       k    = av(kvec)
#       q    = kmax*x(1) + kmin*(1.d0 - x(1))
#       pmin = max(q, abs(k - q))
#       pmax = min(kmax, k + q)
#       if (pmin < pmax) then
#         p    = pmax*x(2) + pmin*(1.d0 - x(2))
#         rcos = (k**2 - q**2 - p**2)/(2.d0*p*q)
#         f2   = 5.d0/7._dp + 0.5_dp*rcos*(p/q + q/p) + 2._dp/7._dp*rcos**2
#         y    = Matterpowerat(p)*Matterpowerat(q)
#         f(1) = 4.d0*pi/k*q*p*y*2.d0*f2**2*(kmax-kmin)*(pmax-pmin)
#       else
#         f(1) = 0.d0
#       endif
#     end subroutine ps22_tf_int

# In[24]:

kmax=1.0
kmin=0.0


# In[28]:

cuba('Cuhre',3,1,gauss_int,.001,0)


# In[29]:

res


# # Binnning Correction

# In[11]:

def gauss_int(ndim, xx, ncomp, ff,userdata):
    x=np.array([xx[i] for i in range(ndim.contents.value)])
    a = np.array([1.,1.,1.])*10**-4
    b = np.array([1.,1.,1.])*10**5
    x=a+(b-a)*x #scale the coordinates
    result=np.exp(-np.dot(x,x)/2.)/(np.sqrt(2.0*np.pi)**3)
    jacobian=np.prod(b-a)*8
    result*=jacobian
    ff[0] = result 
    return 0


# In[217]:




# In[370]:




# In[107]:




# In[ ]:




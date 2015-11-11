import numpy as np
import os.path
from scipy.io import FortranFile
import pyfftw
#import multiprocessing as mp
#import fortran_lib

class Bisp:
      def __init__(self,filename,Ngrid,Ncut,Nmax,Step,FileType):
            """
            Bisp(filename,Ngrid,Ncut,Nmax,Step,FileType)
            
            Bispectrum and Power Spectrum Estimator Class. Ngrid is the 
            side length of the box. Ncut is the minimum kbin, 
            kmin=kf*Ncut, and Nmax is the maximum kbin, kmax=Nmax*kf. 
            Step is an integer which determines the bin size, 
            in units of kf. deltaK=step*kf
            
            FileType is 'F' or 'P' as currently implemented, 
            corresponding to an
                  
                        Ngrid/2+1 x Ngrid x Ngrid 
                        
            Fortran unformatted file -- only possible if scipy.io is 
            installed, or a python pickle file.
            
            It is important to note the the major portion of memory is
            the object:
            
                        phiKshellX(kbins,Ngrid,Ngrid,Ngrid)
                        
            which is a double-valued sequence of real-space "shell-transformed"
            fields, used for efficiently calculating the Bispectrum and
            Power spectrum -- when the underlying field is separable.
            """
            self.FFTW_installed=True
            self.cpus=1
            
            self.VolCalc=False
            self.infile=filename
            self.Ngrid=Ngrid
            self.Ncut=Ncut
            self.Nmax=Nmax
            self.step=Step
            self.FileType=FileType

            # Pad for zeroth Power, zeroth entry
            self.Nk=np.zeros((self.Nmax-self.Ncut)/self.step+2)
            self.pow=np.zeros((self.Nmax-self.Ncut)/self.step+2)
            
            self.Poutfile='pow'+self.tail()
            self.outfile='bisp'+self.tail()
            self.countfile='/Users/rspeare/Data/VB_counts/'+'VB'+self.tail()

            if (self.cpus>1):
                  self.phiKshellXbase=mp.Array('f',(self.Nmax//self.step+1)*self.Ngrid**3)
                  self.phiKshellX=np.ctypeslib.as_array(self.phiKshellXbase.get_obj())
                  self.phiKshellX=self.phiKshellX.reshape(self.Nmax//self.step+1,self.Ngrid,self.Ngrid,self.Ngrid)
                  
                  self.phiKshellX=np.zeros((self.Nmax//self.step+1,self.Ngrid,self.Ngrid,self.Ngrid),dtype=float) #default double prec
            else:
                  self.phiKshellX=np.zeros((self.Nmax//self.step+1,self.Ngrid,self.Ngrid,self.Ngrid),dtype=float) #default double prec
            
            self.bisp=np.zeros((self.Nmax//self.step+1,self.Nmax//self.step+1,self.Nmax//self.step+1),dtype=float) #default double prec

            if( (not self.VolCalc) and (not os.path.exists(self.countfile))):
                  self.VolCalc=True
                  print('Sorry, the Normalization file DNE. Will calculate VB first.')
                  print('Please re-run the once finished')

      def tail(self):
            return '_Ngrid'+str(self.Ngrid)+'_Ncut'+str(self.Ncut)+'_Nmax'+str(self.Nmax)+'_step'+str(self.step)

      def assertNyquistReal(self,dcr,Ngrid):
            """
            Set all Nyquest entries in an FFT k-space field (N/2+1,N,N)
            to be REAL.
            """
            hg=Ngrid//2
            dcr[hg,0,0]=np.real(dcr[0,hg,0])
            dcr[0,hg,0]=np.real(dcr[0,hg,0])
            dcr[0,0,hg]=np.real(dcr[0,0,hg])
            dcr[0,hg,hg]=np.real(dcr[0,hg,hg])
            dcr[hg,hg,hg]=np.real(dcr[hg,hg,hg])
            dcr[hg,0,hg]=np.real(dcr[hg,0,hg])
            dcr[hg,hg,0]=np.real(dcr[hg,hg,0])

      def reflectField(self,field,Ngrid):
            """
            Take an FFT half-field, (N/2+1,N,N) and reflect across the
            x-axis to create a full, physical field, phi(N,N,N)
            """
            phi=np.zeros((Ngrid,Ngrid,Ngrid),dtype=np.complex64)
            phi[:Ngrid//2+1,:,:]=field[:,:,:]
            # REFLECT THE FIELD ACROSS YZ AXIS -- excluding kx or ky or kz = 0
            phi[:Ngrid//2:-1,Ngrid:0:-1,Ngrid:0:-1]=np.conj(phi[1:Ngrid//2,1:Ngrid,1:Ngrid])
            # 2 D reflections to pick up the ``cross hair''
            phi[:Ngrid//2:-1,0,Ngrid:0:-1]=np.conj(phi[1:Ngrid//2,0,1:Ngrid])
            phi[:Ngrid//2:-1,Ngrid:0:-1,0]=np.conj(phi[1:Ngrid//2,1:Ngrid,0])
            # reflect the x-axis
            phi[:Ngrid//2:-1,0,0]=np.conj(phi[1:Ngrid//2,0,0])
            return phi

      def read_fortran_FFTfield(self):
            """
            Read a fortran binary file from FFTW assert all Nyquist
            entries to be real.
            """
            f=FortranFile(self.infile,'r')
            Ng=f.read_ints(dtype=np.int32)[0]
            print('Fortran file Ngrid='+str(Ng))
            if (Ng != self.Ngrid):
                  print('Ngrid values are not equal!')
            dcr=f.read_reals(dtype=np.complex64)
            dcr=np.reshape(dcr,(Ng//2+1,Ng,Ng),order='F')
            return dcr

      def read_python_FFTfield(self):
            """
            Read a pickled FFTW file and assert all Nyquist
            entries to be real.
            """
            dcr=np.load(self.infile)
            Ng=dcr.shape[1]
#            print('ngrid='+str(Ng))
            if (Ng != self.Ngrid):
                  print('Ngrid set='+str(self.Ngrid))
                  print('Ngrid read='+str(Ng))
                  exit()
            # Set NYQUIST to be real
            return dcr

      def shellBin(self,phi):
            """ 
            Bin the k-space density field, and return the array phiKshellK,
            which has shape:
            
            (kbins x Ngrid x Ngrid x Ngrid)
            """
            # FFT convention: array of |kx| values #
            a= np.array([ min(i,self.Ngrid-i) for i in range(self.Ngrid)])
#            a= np.abs(np.fft.fftfreq(self.Ngrid,d=1/self.Ngrid)).astype(int)
            # FFT convention: rank three field of |r| values #
            rk=((np.sqrt(a[:,None,None]**2+a[None,:,None]**2+a[None,None,:]**2)))
            
            irk=(rk/self.step+0.5).astype(int) # BINNING OPERATION, VERY IMPORTANT

            self.Nk=np.array([len(np.where(irk==i)[0]) for i in np.arange(0,(self.Nmax-self.Ncut)/self.step+2)])
            
            phiKshellK=np.zeros((self.Nmax//self.step+1,self.Ngrid,self.Ngrid,self.Ngrid),dtype=complex)
            for i in range(self.Ngrid):
                  for j in range(self.Ngrid):
                        for l in range(self.Ngrid):
                              ak=int(rk[i,j,l]/self.step+0.5) # binning operation
                              if (ak<= self.Nmax/self.step):
                                    phiKshellK[ak,i,j,l]=phi[i,j,l]
            return phiKshellK

      def shellFields(self):
            """
            Create the binned k-shell fields, phi_k(x) using the following steps
            (1) Read phi(k) and complete the field, asserting all Nyquist entries to be real
            (2) Calculate k-shell Maps, such that all entries falling in shell bins are grouped
                  together.
            (3) FFT the shelled Fields: $\phi_k(\vec{x})=\int d^3k\  e^{-ikx}\phi_k(\vec{k})$
            """
            if(not self.VolCalc):
                  if (self.FileType == 'P'):
                        field=self.read_python_FFTfield()
                  elif (self.FileType == 'F'):
                        field=self.read_fortran_FFTfield()
                  else:
                        print('Unknown FileType entered:'+str(self.FileType))
                        return 0

                  self.assertNyquistReal(field,self.Ngrid)
                  phi=self.reflectField(field,self.Ngrid)
            else:
                  phi=np.ones((self.Ngrid,self.Ngrid,self.Ngrid))

            phiKshellK= self.shellBin(phi)

            if (self.FFTW_installed):
                  tempK=pyfftw.n_byte_align_empty((self.Ngrid,self.Ngrid,self.Ngrid),16, dtype='complex64')
                  tempK=phiKshellK[self.Nmax//self.step//2,:,:,:]
                  fftw_ob=pyfftw.builders.fftn(tempK)
                  pyfftw.interfaces.cache.enable()

            # forward FFTW is backward Cosmology FFT
            if (self.FFTW_installed):
                  for j in range(self.Ncut//self.step,self.Nmax//self.step+1):
                        tempK=phiKshellK[j,:,:,:]
                        self.phiKshellX[j]=np.real(fftw_ob(tempK))
                  print(self.phiKshellX.flags)
            else:
                  for j in range(self.Ncut//self.step,self.Nmax//self.step+1):
                        tempK=phiKshellK[j,:,:,:]
                        self.phiKshellX[j]=np.fft.fftn(tempK)
                  print(self.phiKshellX.flags)
      # Monopole power Phat(k)=kF^{-3}P(k). ~3.8 seconds
      def getPow(self):
            """
                  Calculate the  Power Spectrum. Requires shellFields
                  to already have been called on the phiKshellX array
                  to have been initialized. Routine should only
                  take a few seconds.
            """
            for i in np.arange(0,(self.Nmax-self.Ncut)/self.step+2):
                  self.pow[i]=np.einsum('i,i',self.phiKshellX[i].ravel(),self.phiKshellX[i].ravel())/(self.Ngrid**3)/self.Nk[i]# 10 ms

#      @numba.jit
      def getBisp(self):
            """
            For all k1,k2,k3, perform the following real space sum:
            $B(k_{123})=\langle \phi(k_1)\phi(k_2)\phi(k_3) \rangle = \sum_{x}\phi_{k_1}(x)\ \phi_{k_2}(x)\ \phi_{k_3}(x)$
            """
            if (self.cpus>1):
                  z=[]
                  ijl=[[[z.append([i,j,l]) for l in range(max(i-j,b.Ncut//b.step),j+1)] for j in range(b.Ncut//b.step,i+1)] for i in range(b.Ncut//b.step,b.Nmax//b.step+1)];
                  ijl=np.array(z)# physical triangles
                  np.random.shuffle(ijl)

                  num_threads=self.cpus
                  pool=mp.Pool(processes=num_threads)
#                  j1list=np.arange(self.Ncut//self.step,self.Nmax//self.step+1).astype(int)
#                  np.random.shuffle(j1list)
#                  tasks=np.split(j1list,num_threads)
                  tasks=np.array_split(ijl,num_threads)

                  shellSumP=partial(shellEinsum_ijl,self.phiKshellX,self.bisp)
                  pool.map(shellSumP,tasks)                  
                  pool.close()
                  pool.terminate()
                  pool.join()
            else:
                  [self.shellEinsum_i(x) for x in range(self.Ncut//self.step,self.Nmax//self.step+1)]
#                  for i in range(self.Ncut//self.step,self.Nmax//self.step+1):
#                        self.shellEinsum_i(i)

            # If not a Volume Calculation, Normalize the Bispectrum
            if (not self.VolCalc):
                  counts=np.load(self.countfile)
                  counts[np.where(counts==0)]=np.inf
                  self.bisp=self.bisp/counts

#      @numba.jit
      def shellSum_i(self,i):
            """
            For a single k1; all k2,k3, perform the following real space sum:
            $B(k_{123})=\langle \phi(k_1)\phi(k_2)\phi(k_3) \rangle = \sum_{x}\phi_{k_1}(x)\ \phi_{k_2}(x)\ \phi_{k_3}(x)$
            """
            for j in range(self.Ncut//self.step,i+1):
                  for l in range(max(i-j,self.Ncut//self.step),j+1):
                        self.bisp[i,j,l]=(self.phiKshellX[i]*self.phiKshellX[j]*self.phiKshellX[l]).sum() #~116 ms

            if (not self.VolCalc):
                  counts=np.load(self.countfile)
                  counts[np.where(counts==0)]=np.inf
                  self.bisp[i,:,:]=self.bisp[i,:,:]/counts[i,:,:]

#      @numba.jit
      def shellEinsum_i(self,i):
            """
            For a single k1; all k2,k3, perform the following real space sum:
            $B(k_{123})=\langle \phi(k_1)\phi(k_2)\phi(k_3) \rangle = \sum_{x}\phi_{k_1}(x)\ \phi_{k_2}(x)\ \phi_{k_3}(x)$
            
            """
            for j in range(self.Ncut//self.step,i+1):
                  for l in range(max(i-j,self.Ncut//self.step),j+1):
                        self.bisp[i,j,l]=np.einsum('i,i,i',self.phiKshellX[i].ravel(),self.phiKshellX[j].ravel(),self.phiKshellX[l].ravel())


      def export(self):
            if (not self.VolCalc):
                  self.bisp.dump(self.outfile)
            else:
                  self.bisp.dump(self.countfile)
#
def main():
     
      b=BispEstimator()
      print('Ngrid='+str(b.Ngrid))
      print('Ncut='+str(b.Ncut))
      print('Nmax='+str(b.Nmax))
      print('step='+str(b.step))
      print('infile= '+b.infile)
      print('VolumeCalculation= '+str(b.VolCalc))

      print('creating fields...')
      b.shellFields()
      print('summing shells...')
      b.shellSum()
      b.export()

if  __name__ =='__main__':
      main()






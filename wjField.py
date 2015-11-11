import numpy as np
import util
import matplotlib.pyplot as plt

class wjField:
    def __init__(self,step,Ncut,Nmax,Ngrid,kern,win,bound):
        """
        w_j Field Generator. This module convolves a window
        function file of size:
        
        Ngrid x Ngrid x Ngrid

        with a Kernel of your choice, and saves the resulting 
        files for Power Spectrum -- soon Bispectrum -- Interpolation.
        
        --------------------------------------------------------
        Inputs:
        --------------------------------------------------------
        step: step size, in units of k_fun, of the Bispectrum and Power Spectrum evaluation. 
              This specifies the physical bin width.
        Ncut: Lowest kvalue, in units of k_fun, in which to evaluate the P(k) and B(k1,k2,k3)
        Nmax: Highest kvalue, in units of k_fun, in which to evaluate the P(k) and B(k1,k2,k3)
        Ngrid: Size of the Window function array and wjfield array.         
        kern: A string containin the name of the kernel to use, 'gaussian' or 'linear'
        wind: A string, which specifies the name of the window function to use
        
        --------------------------------------------------------
        attributes:
        --------------------------------------------------------
        kx : An Ngrid-long vector, carrying the physical value of kx at each index. (in units of k_f)
        
        p: An Ngrid**3 rank three tensor -- or "field" -- carrying the physical radial value at that
            triple index. (in units of k_f)
            
        pbinned: An Ngrid**3 rank three tensor -- or "field" -- carrying the shell bin indices for every
                triple index [i,j,l]. The shell bins are of size "step". Starting from zero.
        bound: a Logical variable used to determine whether or not to make "hard" cutoffs
               in the Kernel domain. For instance, below jmin*kf and above jmax*kf, should the
               kernels be set explicitly to zero? If so, opt for "true".

        """
        self.Length=2400.0 # Length of the survey
        self.step=step
        self.Ngrid=Ngrid
        self.Nmax=Nmax
        self.Ncut=Ncut
        self.jmax=self.Nmax//self.step
        self.jmin=self.Ncut//self.step
        self.kx= np.fft.fftfreq(self.Ngrid,d=1./240.).astype(float)
        self.p=((np.sqrt(self.kx[:,None,None]**2+self.kx[None,:,None]**2+self.kx[None,None,:]**2))).astype(float)
        self.pbinned=self.bin_radius(self.p)
        self.Nk=np.bincount(self.pbinned.ravel()) # number of grid cells in each shell bin


        # Set Kernels to zero outside kmin and kmax
        self.bound=bound
        print('Boundary='+str(self.Boundary))

        self.sortedindex=np.argsort(self.pbinned.ravel())

        # Initialize What Kernel and What window you'd like to use
        Windows=['SDSSII','Dirac','Gauss','sphere','sphere_simple']
        self.Window=win
        if (win not in Windows):
            print('invalid window chosen')

        Kernels=['linear','Gaussian']
        self.kernel=kern
        if (kern not in Kernels):
            print('invalid kernel chosen')

        self.sigma=0.0
        if (kern == Kernels[1]):
            self.sigma=1.0
        self.create_Window()

        self.Gname='Gmatrix_'+str(self.bound)+'_Ngrid_'+str(self.Ngrid)+'_'+str(self.jmax)+'_'+str(self.sigma)+'_'+self.kernel+'_'+self.Window+'.pickle'
    def A(self,p,j):
        """
        The set of Kernels, A_j(p) in k-space. 
        """ 
        if (self.kernel =='linear'):
            x=np.abs((p-j*self.step)/self.step)
            return self.TopHat(x)*(1.-x)*self.Boundary(p)
        elif (self.kernel == 'Gaussian'):
            x=(p-j*self.step)/self.step
            return self.Gaussian(x)*self.Boundary(p)
        else:
            print('invalid kernel entered')

    def Gaussian(self,k):
        """
        Return exp(-k*k/(2*sigma**2)).
        Where sigma is in units of step size k_fundamental.
       
        sigma=sigma*step
        """
        return np.exp(-k**2/(2.0*self.sigma**2))
    def TopHat(self,x):
        """
        return 1 if -1<x<1; 0 otherwise.
        """
        return np.where(np.abs(x)<=1.,1.,0.)

    def create_pvec(self):
        """
        Return an array of size
        
        Ngrid x Ngrid x Ngrid x 3
        
        Where each entry is: kx,ky,kz
        """
        rx=np.fft.fftfreq(self.Ngrid,1./self.Ngrid)[:,None,None]
        self.pvec=np.zeros((self.Ngrid,self.Ngrid,self.Ngrid,3))
        for i in np.arange(self.Ngrid):
            for j in np.arange(self.Ngrid):
                for l in np.arange(self.Ngrid):
                    self.pvec[i,j,l]=[rx[i],rx[j],rx[l]]

        self.p=np.lin
    def vol(self,R):
        """
        Return the number of Grid cells within radius R.
        Not Vectorizable.
        """
        return len(np.where(self.p<=R)[0])
    def WthK(self,x):
        """ 
        Spherical TopHat (3D) in fourier space.
        """
        x=x.astype(float)
        return np.where(np.isclose(x,0.),1.,3.0*(np.sin(x)-x*np.cos(x))/x**3)

    def WthX(self,x,R):
        """
        TopHat Function, with scaled width R and divided by Volume:
        
        Wth(x,R)= TopHat(x/R)/Vol_R
        """
        return np.where(np.abs(x/R)<=1.,1.,0.)/self.vol(R)
    
    def bin_radius(self,r):
          """
                Using the given step size, bin radial value into the
                proper shell. Input (float), output unsigned integer. 
                This function is vectorized.
          """
          return (r/self.step+0.5).astype(np.uint16) # unsigned integer for memory

    def phys_index(self,i):
        """
        Return the physical index of an FFT array
        """
        return ((i+ self.Ngrid/2-1) % self.Ngrid) - self.Ngrid/2+1

    def Boundary(self,x):
        """
        Return 1 if Nmin < x < Nmax, 0 otherwise. 
        This function is vectorized.
        """
        if (self.bound):
            outside=0.
        else:
            outside=1.
        return np.where(x<=self.Nmax,1.,outside)*np.where(x>=self.Ncut,1.,outside)


    def create_Window(self):
        if (self.Window == 'SDSSII'):
            print('creating SDSSII window function')
            if (self.Ngrid==240):
                  path="/Users/rspeare/Data/window/measurements/FFT240-ran.pickle"
            elif (self.Ngrid==480):
                  path="/Users/rspeare/Data/window/measurements/FFT480-ran_v5.pickle"
            else:
                  raise ValueError('invalid Ngrid for the SDSS window function')
            W=np.load(path)
            util.assertNyquistReal(W) # For real fields in x space, assert Hermitian in kspace
            W=util.reflectField(W) # Make array ngrid*ngrid*ngrid
            norm=W[0,0,0]
            Wx=np.fft.fftn(W)
        elif (self.Window=='sphere'):
            Wx=np.zeros((self.Ngrid,self.Ngrid,self.Ngrid),dtype=np.complex64)
            W=np.zeros((self.Ngrid,self.Ngrid,self.Ngrid),dtype=np.complex64)
            Rmax=120.0
            Rmin=46.5
            vRmax=self.vol(Rmax)
            vRmin=self.vol(Rmin)
            W=(vRmax*self.WthK(Rmax*self.p)-vRmin*self.WthK(Rmin*self.p))/(vRmax-vRmin)
            print('possible NaN positions:')
            print(np.where(np.isnan(W)))

            Wx=np.fft.fftn(W)
            norm=W[0,0,0]
        elif (self.Window=='sphere_simple'):
            Wx=np.zeros((self.Ngrid,self.Ngrid,self.Ngrid),dtype=np.complex64)
            W=np.zeros((self.Ngrid,self.Ngrid,self.Ngrid),dtype=np.complex64)
            Rmax=120.0
            Rmin=46.5
            Wx+=self.TopHat(self.p/Rmax)
            Wx-=self.TopHat(self.p/Rmin)
            W=np.fft.ifftn(Wx)
            norm=W[0,0,0]

        print('W[0,0,0] Window Function norm = '+str(norm))            
        if (norm <= 0.):
            print('norm is less than zero, continue?')
#            raise ValueError('WARNING WINDOW FUNCTION NORMALIZATION IS LESS THAN ZERO. Check to make sure that the grid resolution (self.Ngrid) is high enough!!')
        self.Wx=Wx
#        self.W=W


    def shell_avg(self,phi):
        """
        Shell average the field phi, which has dimension Ngrid x Ngrid x Ngrid, 
        into the radial shells corresponding to jmin --> jmax in bin sizes of 
        self.step. 
        """
        sortedsum=phi.ravel()[self.sortedindex]
        ind=np.cumsum(self.Nk)
        ShellAvg=np.zeros(len(self.Nk)).astype(np.complex64)
        for nn in np.arange(1,self.jmax+1):
            first,last=ind[nn-1],ind[nn]
            ShellAvg[nn]=sortedsum[first:last].mean()
        return ShellAvg

    def create_wj_pow(self,*kwargs):
        """ 
        This loop goes from jmin to jmax, creating all w_j(q) by convolving
        our Kernel with the window function. For the power spectrum case, 
        we need to convolve the kernels with the square modulus of the 
        window function. 

        RETURNS CONVOLUTION OF THE MODULUS, what we might call -- for the
        power spectrum -- the w_j fields:

        w_j(k) = \int dq A_j(q) W(k-q)^2

        """
        self.G=np.zeros((self.jmax-self.jmin+1,self.jmax-self.jmin+1))
        W=np.fft.ifftn(self.Wx)
        Wxconvolved=np.fft.fftn(np.abs(W)**2)
        print('creating wj_Fields for Power Spectrum...')
        for j in np.arange(self.jmin,self.jmax+1):
            print(j)
            tempK=self.A(self.p,j).astype(np.complex64)
            wjpow=np.fft.ifftn(np.fft.fftn(tempK)*Wxconvolved)
            self.G[:,j-self.jmin]=self.shell_avg(wjpow)[self.jmin:self.jmax+1]

    def shellBin(self,phi):
        """ 
        Bin the k-space density field, and return the array phiKshellK,
        which has shape:
        
        (kbins x Ngrid x Ngrid x Ngrid)
        """
        phiKshellK=np.zeros((len(self.Nk),self.Ngrid,self.Ngrid,self.Ngrid),dtype=complex)
        for i in range(self.Ngrid):
            for j in range(self.Ngrid):
                for l in range(self.Ngrid):
                    ak=self.pbinned[i,j,l]
                    if (ak<= self.jmax):
                        phiKshellK[ak,i,j,l]=phi[i,j,l]
        return phiKshellK

    def plot_Kernels(self):
        lim=np.abs(np.amin(self.kx))
        jj=(self.jmin+self.jmax)//2
        offset=10

        a = np.linspace(start=-lim, stop=lim, num=200)
        xx=((np.sqrt(a[:,None]**2+a[None,:]**2)))
        plt.figure(figsize=(12,12))
        ax1=plt.subplot2grid((3,2),(0,0))
#        ax1.imshow(self.Gaussian(xx,1.), extent=[-lim,lim,-lim,lim])
        ax1.imshow(np.abs(np.fft.fftshift(np.fft.fftn(self.Wx)[offset],axes=(0,1))),interpolation='none',extent=[-lim,lim,-lim,lim])
        ax1.set_title(str(self.Window)+' Window Function, W(k)')
        ax2=plt.subplot2grid((3,2),(2,1))
#        ax2.imshow(self.A(xx,jj), extent=[-lim,lim,-lim,lim])
        ax2.imshow(self.A(np.fft.fftshift(self.p[0],axes=(0,1)),jj),extent=[-lim,lim,-lim,lim])

        ax2.set_title('Kernel $A(p,j)$   $j=$'+str(jj))
        ax3=plt.subplot2grid((3,2),(1,0))
#        ax3.imshow(self.Boundary(xx), extent=[-lim,lim,-lim,lim])
        ax3.imshow(self.Boundary(np.fft.fftshift(self.p[0],axes=(0,1))),extent=[-lim,lim,-lim,lim])
        ax3.set_title('Kernel Boundary $ A(p,j)$')
        ax4=plt.subplot2grid((3,2),(1,1))
        plist=np.linspace(self.Nmax-2*self.step,self.Nmax+0.5,500)
        ax4.plot(plist,self.A(plist,self.jmax))
        ax4.plot(plist,self.A(plist,self.jmax-1))
        ax4.plot(plist,self.A(plist,self.jmax-2))
        ax4.set_title('Kernels $A_{j},A_{j-1}, A_{j-2}$ at Boundary')
        ax5=plt.subplot2grid((3,2),(2,0))
        ax5.imshow(self.p[0])
        ax5.imshow(np.abs(np.fft.fftshift(self.p[0],axes=(0,1))),interpolation='none',extent=[-lim,lim,-lim,lim])
        ax5.set_title('p field')
        ax6=plt.subplot2grid((3,2),(0,1))
        ax6.imshow(np.fft.fftshift(np.real(self.Wx[0]).astype(float),axes=(0,1)),interpolation='none',extent=[-lim,lim,-lim,lim])
#        ax6.imshow(np.abs(np.fft.fftshift(self.pbinned[0],axes=(0,1))),interpolation='none',extent=[-lim,lim,-lim,lim])
        ax6.set_title('W(x) Window Function')
        plt.show()

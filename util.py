import numpy as np
import os.path
from scipy.io import FortranFile
import triangle
import matplotlib.pyplot as plt

def export_FFT_field(phi,outfile):
    """
    export_FFT_field(phi,outfile)

    outfile: string
    phi[:,:,:] : np.array, rank 3

    This function takes in a 3-D field, of dimension
    Ngrid x Ngrid x Ngrid, and exports a half-field
    Ngrid/2+1 x Ngrid x Ngrid in unformatted
    Fortran record. Equivalent to a write(field) statement.
    """
    f=FortranFile(outfile,'w')
    n=phi.shape[2]
    f.write_record(np.array([n],dtype=np.int32)) # write integer record
    f.write_record(phi[:n//2+1].ravel(order='F')) # write half-field
    f.close()

def read_python_FFTfield(infile):
    """
    Read in a python pickle field
    """
    return np.load(infile)

def shell_avg2(self,phi):
    sortedsum=phi.ravel()[self.sortedindex]
    ind=np.cumsum(self.Nk)
    ShellAvg=np.zeros(len(self.Nk)).astype(np.complex64)
    for nn in np.arange(1,self.jmax+1):
        first,last=ind[nn-1],ind[nn]
        # not sure if this is going to work, but for low volume
        # shells, use the median as the estimator of the mean
        # in order to protect against fat tails
        #            if (first-last <= 100):
        #                ShellAvg[nn]=np.median(sortedsum[first:last])
        #            else:
        ShellAvg[nn]=sortedsum[first:last].mean()
        return (ShellAvg)[self.jmin:self.jmax+1]
#        return ShellAvg    
def read_fortran_FFTfield(infile):
    """
    Read a Half-Field with FFTW indexing from
    a Fortran Unformatted Binary file. The first
    entry is a single integer.
    """
    f=FortranFile(infile,'r')
    Ngrid=f.read_ints(dtype=np.int32)[0]
    print('Fortran file Ngrid='+str(Ngrid))
    dcr=f.read_reals(dtype=np.complex64)
    dcr=np.reshape(dcr,(Ngrid//2+1,Ngrid,Ngrid),order='F')
    dcr.dump(infile+'.pickle') # Save infile as a pickle
    return dcr
    
def reflectField(field):
    """
    Take an FFT half-field, field(N/2+1,N,N) and reflect across the
    x-axis to create a full, physical field, phi(N,N,N)
    field(N/2+1,N,N) ---> phi(N,N,N)
    """
    Ngrid=field.shape[1]
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

def assertNyquistReal(dcr):
    """
    Set all Nyquest entries in an FFT k-space field (N/2+1,N,N)
    to be REAL.
    """
    Ngrid=dcr.shape[1]
    hg=Ngrid//2
    dcr[hg,0,0]=np.real(dcr[0,hg,0])
    dcr[0,hg,0]=np.real(dcr[0,hg,0])
    dcr[0,0,hg]=np.real(dcr[0,0,hg])
    dcr[0,hg,hg]=np.real(dcr[0,hg,hg])
    dcr[hg,hg,hg]=np.real(dcr[hg,hg,hg])
    dcr[hg,0,hg]=np.real(dcr[hg,0,hg])
    dcr[hg,hg,0]=np.real(dcr[hg,hg,0])

def phys_index(i,Ng):
    return ((i+ Ng/2-1) % Ng) - Ng/2+1
def comp_phys_index(i,Ng):
    return (( Ng+1-i) % Ng)+2

def wval(j,phi,ixiyiz):
    return np.real(phi[ixiyiz[j,0],ixiyiz[j,1],ixiyiz[j,2]])

def plot_FFT_file(file,critval,part):
    """
    Plot the Fortran FFT file 'file', with a triangle plot, showing all elements
    above critval.
    """
    phi=read_fortran_FFTfield(file)
    phi=reflectField(phi)
    assertNyquistReal(phi)
    phi=phi/np.amax(phi)
    if (part == 'm'):
        phi=np.sqrt(np.conj(phi)*phi)
    elif (part =='r'):
        phi=np.real(phi)
    elif (part=='i'):
        phi=np.imag(phi)
    pos=np.where(np.abs(phi)>critval)
    print(str(len(pos[0]))+' points')
    indices=np.vstack((np.where(phi>critval)[0],np.where(phi>critval)[1],np.where(phi>critval)[2])).T
    sample=np.vstack([phys_index(indices[:,0],240),phys_index(indices[:,1],240),phys_index(indices[:,2],240)]).T
    triangle.corner(sample,labels=["kx","ky","kz"],truths=[0.0, 0.0, 0.0])


def sum_FFT_field(file):
    """
    Sum the Fortran FFT file 'file'. \int W(k) d^3k
    """


    phi=read_fortran_FFTfield(file)
    phi=reflectField(phi)
    assertNyquistReal(phi)
    phi=phi/np.amax(phi)
    return np.sum(phi)

def plot_FFT_file3D(file,critval,part):
    """
    Plot the Fortran FFT file 'file', with a 3D scatter plot showing all elements
    above critval.
    """
    phi=read_fortran_FFTfield(file)
    assertNyquistReal(phi)
    phi=reflectField(phi)
    phi=phi/np.amax(phi)
    if (part == 'm'):
        phi=np.sqrt(np.conj(phi)*phi)
    elif (part =='r'):
        phi=np.real(phi)
    elif (part=='i'):
        phi=np.imag(phi)

    pos=np.where(np.abs(phi)>critval)
    print(str(len(pos[0]))+' points')
    ixiyiz=np.vstack((pos[0],pos[1],pos[2])).T
    colors=np.real(wval(np.arange(len(ixiyiz)),phi,ixiyiz))
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('qx')
    ax.set_ylabel('qy')
    ax.set_zlabel('qz')
    ax.view_init(elev=10.,azim=260)
    sample=np.vstack([phys_index(ixiyiz[:,0],240),phys_index(ixiyiz[:,1],240),phys_index(ixiyiz[:,2],240)]).T
    p=ax.scatter(sample[:,0],sample[:,1],sample[:,2],c=colors,zdir='z')
    fig.colorbar(p,shrink=.4,aspect=20)
    ax.set_title('SDSSII Window Function')    

def plot_FFT_file3D_half(file,critval,part):
    """
    Plot the Fortran FFT file 'file', with 3D scatter plot.
    But only show the Half plane
    """
    phi=read_fortran_FFTfield(file)
    assertNyquistReal(phi)
    phi=phi/np.amax(phi)
    if (part == 'm'):
        phi=np.sqrt(np.conj(phi)*phi)
    elif (part =='r'):
        phi=np.real(phi)
    elif (part=='i'):
        phi=np.imag(phi)

    pos=np.where(np.abs(phi)>critval)
    print(str(len(pos[0]))+' points')
    ixiyiz=np.vstack((pos[0],pos[1],pos[2])).T
    colors=np.real(wval(np.arange(len(ixiyiz)),phi,ixiyiz))
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('qx')
    ax.set_ylabel('qy')
    ax.set_zlabel('qz')
    ax.view_init(elev=10.,azim=260)
    sample=np.vstack([phys_index(ixiyiz[:,0],240),phys_index(ixiyiz[:,1],240),phys_index(ixiyiz[:,2],240)]).T
    p=ax.scatter(sample[:,0],sample[:,1],sample[:,2],c=colors,zdir='z')
    fig.colorbar(p,shrink=.4,aspect=20)
    ax.set_title('SDSSII Window Function')    

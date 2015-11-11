import numpy as np
#import numba

class regressor:
      def __init__(self,X,T,*args,**kwargs):
            """
                  regressor(Xtrain, Ytrain, kernel='name', C='ll',theta='theta')
                  
                  kernel is a string: 'linear' or 'gaussian' or 'watson'
                  
                  lamb is float: regularization parameter lambda, essentially
                        the inverse variance of the prior on Kernel weights w.
                  
                  theta is vector of floats: theta, sigma

                  The amplitude and scale of the Gaussian kernel
                  Creates the Regressor type object. kernel is 
                  the choice of kernel you'd like to use 
                  for regression. lambda is the regularization 
                  parameter.
            """
            self.X=X
            self.T=T
            self.deltaX=X[1]-X[0]
            #            self.N=len(self.X)
            if (len(X.shape)==1):
                  print('scalar input')
                  self.N=X.shape[0]
                  self.D=1
            else:
                  self.N,self.D= X.shape
                  print('vector input')

############### KERNEL SELECTION ##################
            kernel=kwargs.get('kernel',None)
            if (kernel == 'gaussian'):
                  self.kernel = self.kernelG
            elif (kernel == 'linear'):
                  self.kernel = self.kernelTri
            elif (kernel == 'exponential'):
                  self.kernel=self.kernelExp
            elif (kernel == None):
                  print('no kernel entered. Defaulting to Linear')
                  self.kernel = self.kernelG
            else:
                  print('invalid kernel entered')
############### REGULARIZATION ##################
            ll=kwargs.get('C',None)
            if ( ll == None ):
                  print('no lambda entered. Defaulting to 1.e-6')
                  self.ll=10.**(-6)
            else:
                  self.ll=ll

############### HYPER PARAMETER SELECTION ##################
            theta=kwargs.get('theta',None)
            if (self.kernel == self.kernelG and theta==None):
                  print('no variance entered. Defaulting to twice input spacing')
                  
                  # Set the length scale of the RBF Gaussian kernel to be twice
                  # the spacing between input points
                  self.sigma=(self.X[1]-self.X[2])*2.
                  
                  # Set the variance of the Gaussian process to be the minimum of
                  # the regressed function.
                  self.theta=np.amin(np.abs(self.T))
            else:
                  if (len(theta)==2):
                        self.sigma=theta[1]
                        self.theta=theta[0]

############### DATA VARIANCE ##################
            self.noise=kwargs.get('noise',np.zeros(len(self.X)))

############### CREATE INVERSE KERNEL MATRIX ##################
            self.fit()

      
      #numba.jit
      def kernelExp(self,x1,x2):
            """
                  Gaussian Kernel. Can take in a list of vectors
                  and output:
                  
                  exp(-x**2/2)
                  
                  for each vector, using the Euclidean norm.
                  
                  Currently vectorized for scalar input
                  """
            if (self.D !=1):
                  z=np.linalg.norm(x1-x2,axis=1)/self.sigma
                  return np.exp(-z)*self.theta
            else:
                  if hasattr(x1,"__len__"):
                        z=(x1.reshape((len(x1),1))-x2.reshape((1,len(x2))))/self.sigma
                  else:
                        z=(x1-x2)/self.sigma
                  return np.exp(-np.abs(z))*self.theta
      def kernelG(self,x1,x2):
            """
                  Gaussian Kernel. Can take in a list of vectors
                  and output:
                  
                  exp(-x**2/2)
                  
                  for each vector, using the Euclidean norm.
                  
                  Currently vectorized for scalar input
            """
            if (self.D !=1):
                  z=np.linalg.norm(x1-x2,axis=1)/self.sigma
                  return np.exp(-z**2/2.)*self.theta
            else:
                  if hasattr(x1,"__len__"):
                        z=(x1.reshape((len(x1),1))-x2.reshape((1,len(x2))))/self.sigma
                  else:
                        z=(x1-x2)/self.sigma
                  return np.exp(-z**2/2.)*self.theta
      #numba.jit
      def kernelTri(self,x1,x2):
            """
                  linear Kernel. Can take in a list of vectors
                  and output:
                  
                  (1-abs(x)) for  -1 < x < 1, else 0
                  
                  for each vector, using the Euclidean norm.
            """
            # If input is vector, sum over D index to return list of absolute
            # distances
            if (self.D !=1):
                  x=np.linalg.norm(x1-x2,axis=1)
            else:
                  # If input is a scalar, check for vector input.
                  if hasattr(x1,"__len__"):
                        x=np.abs((x1.reshape((len(x1),1))-x2.reshape((1,len(x2)))))
                  else:
                        x=np.abs(x1-x2)
            x=x/self.deltaX
            return np.where(x<=1,(1.-x),0)
      #numba.jit
      
      def kvec(self,x):
            """
            Returns k(x,X), the Kernel Vector a vector of length len(X)=self.N
            
            """
            return self.kernel(x,self.X)

      #numba.jit
      def create_Gram_Matrix(self):
            """
            Initializes K(x_m, x_n) matrix. Vectorized row
            implementation using kvec(x)
            """
            self.K=np.zeros((self.N,self.N)).astype(float)
            for i in np.arange(self.N):
                  self.K[i]=self.kvec(self.X[i])
      #numba.jit

      def fit(self):
            """
            Fit to the data X,Y
            """
            self.create_Gram_Matrix()
            try:
                  self.L=np.linalg.cholesky(self.K + np.eye(len(self.K))*self.noise)
                  self.Linv=np.linalg.solve(self.L,np.eye(len(self.K)))
                  self.alpha=np.dot(self.Linv.T,np.dot(self.Linv,self.T))
            except :
                  print('Cholesky Decompositon unstable, doing explicit inversion of Gram')
                  self.invert_Gram_Matrix()


      def invert_Gram_Matrix(self):
            """
                  Invert the regularized gram matrix, (K+lambda)
                  using np.linalg.solve
            """
            if (any(self.noise == 0)):
                  self.Kinv=np.linalg.solve(self.K+np.eye(self.N)*self.noise**2,np.eye(self.N))
            else:
                  self.Kinv=np.linalg.solve(self.K+self.ll*np.eye(self.N),np.eye(self.N))
            if (self.kernel=='linear'):
                  self.model=self.T
#                  self.model=np.dot(self.Kinv,self.T)
            else:
                  self.alpha=np.dot(self.Kinv,self.T)
      #numba.jit
      def predict(self,x):
            mean=np.dot(self.kvec(x),self.alpha)
            #            if hasattr(x,'__len__'):
            #                  v=np.array([np.dot(self.Linv,self.kvec(xx)) for xx in x])
            
            #            v=np.dot(self.Linv,self.kvec(x).T)
            #            if hasattr(x,'__len__'):
            #      var=self.kernel(x,x).diagonal()-np.dot(v.T,v).diagonal()
            #else:
            #      var=self.kernel(x,x)-np.dot(v.T,v)


            return mean







from __future__ import print_function, division
import numpy as np
from scipy import linalg

from smt.methods import LS, PA2, KPLS, KPLSK, GEKPLS, KRG
from smt.problems import Sphere
from smt.sampling import LHS

# Initialization of the problem
ndim = 10
ndoe = int(10*ndim)

# Define the function
fun = Sphere(ndim = ndim)

# Construction of the DOE
sampling = LHS(xlimits=fun.xlimits,criterion = 'm')
xt = sampling(ndoe)

# Compute the output
yt = fun(xt)
# Compute the gradient
for i in range(ndim):
    yd = fun(xt,kx=i)
    yt = np.concatenate((yt,yd),axis=1)

# Construction of the validation points
ntest = 500
sampling = LHS(xlimits=fun.xlimits)
xtest = sampling(ntest)
ytest = fun(xtest)

########### The LS model

# Initialization of the model
t = LS()
# Add the DOE
t.add_training_points('exact',xt,yt[:,0])
# Train the model
t.train()
# Prediction of the validation points
y = t.predict_value(xtest)

# Prediction of the derivatives with regards to each direction space
print("***************************************************************")
print("***Prediction derivatives***")
print("***************************************************************")
yd_prediction = np.zeros((ntest,ndim))
for i in range(ndim):
    yd_prediction[:,i] = t.predict_derivative(xtest,kx=i).T

print('LS,  err: '+str(linalg.norm(y.reshape((ntest,1))-ytest.reshape((ntest,
            1)))/linalg.norm(ytest.reshape((ntest,1)))))


########### The PA2 model
t = PA2()
t.add_training_points('exact',xt,yt[:,0])
t.train()
y = t.predict_value(xtest)

# Prediction of the derivatives with regards to each direction space
print("***************************************************************")
print("***Prediction derivatives***")
print("***************************************************************")
yd_prediction = np.zeros((ntest,ndim))
for i in range(ndim):
    yd_prediction[:,i] = t.predict_derivative(xtest,kx=i).T

print('PA2,  err: '+str(linalg.norm(y.reshape((ntest,1))-ytest.reshape((ntest,
            1)))/linalg.norm(ytest.reshape((ntest,1)))))

########### The Kriging model
# The variable 'theta0' is a list of length ndim.
t = KRG(theta0=[1e-2]*ndim)
t.add_training_points('exact',xt,yt[:,0])

t.train()
y = t.predict_value(xtest)

print('Kriging,  err: '+str(linalg.norm(y.reshape((ntest,1))-ytest.reshape((ntest,
            1)))/linalg.norm(ytest.reshape((ntest,1)))))

# Prediction of the derivatives with regards to each direction space
print("***************************************************************")
print("***Prediction derivatives***")
print("***************************************************************")
yd_prediction = np.zeros((ntest,ndim))
for i in range(ndim):
    yd_prediction[:,i] = t.predict_derivative(xtest,kx=i).T

# Variability of the model for any x
print("***************************************************************")
print("***Variability of the model***")
print("***************************************************************")
variability = t.predict_variance(xtest)    

########### The KPLS model
# The variables 'name' must be equal to 'KPLS'. 'n_comp' and 'theta0' must be
# an integer in [1,ndim[ and a list of length n_comp, respectively. Here is an
# an example using 1 principal component.
t = KPLS( n_comp=2, theta0=[1e-2,1e-2])
t.add_training_points('exact',xt,yt[:,0])

t.train()
y = t.predict_value(xtest)

print('KPLS,  err: '+str(linalg.norm(y.reshape((ntest,1))-ytest.reshape((ntest,
            1)))/linalg.norm(ytest.reshape((ntest,1)))))

# Prediction of the derivatives with regards to each direction space
print("***************************************************************")
print("***Prediction derivatives***")
print("***************************************************************")
yd_prediction = np.zeros((ntest,ndim))
for i in range(ndim):
    yd_prediction[:,i] = t.predict_derivative(xtest,kx=i).T
    
# Variability of the model for any x
print("***************************************************************")
print("***Variability of the model***")
print("***************************************************************")
variability = t.predict_variance(xtest)    
    
########### The KPLSK model
# 'n_comp' and 'theta0' must be an integer in [1,ndim[ and a list of length n_comp, respectively.
t = KPLSK(n_comp=2, theta0=[1e-2,1e-2])
t.add_training_points('exact',xt,yt[:,0])
t.train()
y = t.predict_value(xtest)

print('KPLSK,  err: '+str(linalg.norm(y.reshape((ntest,1))-ytest.reshape((ntest,
            1)))/linalg.norm(ytest.reshape((ntest,1)))))

# Prediction of the derivatives with regards to each direction space
print("***************************************************************")
print("***Prediction derivatives***")
print("***************************************************************")
yd_prediction = np.zeros((ntest,ndim))
for i in range(ndim):
    yd_prediction[:,i] = t.predict_derivative(xtest,kx=i).T

# Variability of the model for any x
print("***************************************************************")
print("***Variability of the model***")
print("***************************************************************")
variability = t.predict_variance(xtest)    
    
########### The GEKPLS model using 1 approximating points
# 'n_comp' and 'theta0' must be an integer in [1,ndim[ and a list of length n_comp, respectively.
t = GEKPLS(n_comp=1, theta0=[1e-2], xlimits=fun.xlimits,delta_x=1e-4,extra_points= 1)
t.add_training_points('exact',xt,yt[:,0])
# Add the gradient information
for i in range(ndim):
    t.add_training_points('exact',xt,yt[:, 1+i].reshape((yt.shape[0],1)),kx=i)

t.train()
y = t.predict_value(xtest)

print('GEKPLS1,  err: '+str(linalg.norm(y.reshape((ntest,1))-ytest.reshape((ntest,
            1)))/linalg.norm(ytest.reshape((ntest,1)))))

print("***************************************************************")
print("***Prediction derivatives***")
print("***************************************************************")
# Prediction of the derivatives with regards to each direction space
yd_prediction = np.zeros((ntest,ndim))
for i in range(ndim):
    yd_prediction[:,i] = t.predict_derivative(xtest,kx=i).T

# Variability of the model for any x
print("***************************************************************")
print("***Variability of the model***")
print("***************************************************************")
variability = t.predict_variance(xtest)    

########### The GEKPLS model using 2 approximating points
# 'n_comp' and 'theta0' must be an integer in [1,ndim[ and a list of length n_comp, respectively.
t = GEKPLS(n_comp=1, theta0=[1e-2], xlimits=fun.xlimits,delta_x=1e-4,
         extra_points= 2)
t.add_training_points('exact',xt,yt[:,0])
# Add the gradient information
for i in range(ndim):
    t.add_training_points('exact',xt,yt[:, 1+i].reshape((yt.shape[0],1)),kx=i)

t.train()
y = t.predict_value(xtest)

print('GEKPLS2,  err: '+str(linalg.norm(y.reshape((ntest,1))-ytest.reshape((ntest,
            1)))/linalg.norm(ytest.reshape((ntest,1)))))

print("***************************************************************")
print("***Prediction derivatives***")
print("***************************************************************")
# Prediction of the derivatives with regards to each direction space
yd_prediction = np.zeros((ntest,ndim))
for i in range(ndim):
    yd_prediction[:,i] = t.predict_derivative(xtest,kx=i).T

# Variability of the model for any x
print("***************************************************************")
print("***Variability of the model***")
print("***************************************************************")
variability = t.predict_variance(xtest)    
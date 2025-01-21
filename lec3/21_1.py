import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#use pandas to load the real estate dataset
df=pd.read_csv("real_estate_dataset.csv")


#get the number of samples and features
n_samples, n_features=df.shape
#display the first 5 rows of the dataset
print(df.head())

#print the number of samples and features
print("number of samples", "number of features:", n_samples,n_features)

#save the features/colums of the dataset
columns=df.columns
np.savetxt("real_estate_dataset_columns.txt",columns,fmt="%s")


#use the square feet, garage size, loaction score and distance to center as features
X=df[['Square_Feet','Garage_Size','Location_Score','Distance_to_Center']]

#use the price column as the target
y=df['Price']
print(y)

n_samples, n_features=X.shape
#print the number of samples and features
print("number of samples", "number of features:", n_samples,n_features)

#initialising coeffiecients with ones
coefs=np.ones(n_features+1)
# +1 to include bias term

predictions_base=X@coefs[1:]+coefs[0]

#append a column of ones to the input matrix X
X=np.hstack([np.ones((n_samples,1)),X])

#predict the prices for each samples in X
predictions=X@coefs

#checking if predictions_base and predictions are same or not
is_same=np.allclose(predictions_base,predictions)
print("Is predictions_base and predictions same:",is_same)

#calculate the error between the prediction and target
errors=y-predictions
rel_error=errors/y

#calculating loss - brute force way
loss_loop=0
for i in range(n_samples):
    loss_loop+=errors[i]**2

loss_loop/=n_samples

#calculating loss using matrices
loss_matrix=np.transpose(errors)@errors/n_samples

#checking if loss_loop and loss_matrix are same or not
is_diff=np.allclose(loss_loop,loss_matrix)
print("Is the loss calculated using a loop equal to the loss calculated using matrix operations?", is_diff)

print("Size of the errors:", errors.shape)
print("L2 norm of the errors:",np.linalg.norm(errors))
print("Relative error of L2 norm:", np.linalg.norm(rel_error))

#We want to find the coefficients that minimize the loss funtion
#This problem is called least squares problem
#so for a set of coefficients to be solution, the gradient of loss function evaluated wrt coeffs should be zero
#Also the Hessian of the loss function evaluated wrt coeffs should be positive definite

#write the loss_matrix in terms of data and coefficients
loss_matrix=(y-X@coefs).T@(y-X@coefs)/n_samples

#calculate the gradient of the loss funtion wrt the coefficients
grad_matrix= -2/n_samples*X.T@(y-X@coefs)

#set the gradient to zero and solve for the coefficients
coefs_new=np.linalg.inv(X.T@X)@X.T@y

#save the coefficients to a file
np.savetxt("coefs.csv",coefs_new,delimiter=",")

#calculate the prices using the new coefficients
predictions_model=X@coefs_new

#calculate the error using the new coefficients
errors_model=y-predictions_model
rel_error_model=errors_model/y

print("L2 norm of the errors using new coefficients:",np.linalg.norm(errors_model))
print("L2 norm of relative error using new coefficinets:",np.linalg.norm(rel_error_model))



#############################################  QR   ##########################################################

#calculating inverse of X.T@X is the expensive step

#instead of only 4 features we take all features
X=df.drop(columns=['Price'])
n_samples,n_features=X.shape

#print the number of samples and features
print("number of samples", "number of features:", n_samples,n_features)

X=np.hstack((np.ones((n_samples,1)), X))
#QR factorisation of X
Q,R=np.linalg.qr(X)
print("Shape of Q:", Q.shape)
print("shape of R:", R.shape)

#save Q matrix to a file
np.savetxt("Q.csv",Q,delimiter=",")
#save R to a file
np.savetxt("R.csv",R, delimiter=",")

#identity matrix
sol=Q.T@Q 
np.savetxt("sol.csv",sol,delimiter=",")

#X=Q@R
#X.T@X=R.T@Q.T@Q@R=R.T@R
#X.T@y=R.T@Q.T@y
#R@coefs=Q.T@y

# let us define b as following
b=Q.T@y

print("b shape",b.shape)
print("R shape", R.shape)

#R is upper traingular, so coefs can be calculated using backsubstitution
coefs_qr=np.zeros(n_features+1)

for i in range(n_features,-1,-1):
    coefs_qr[i]=b[i]
    for j in range(i+1,n_features+1):
        coefs_qr[i]-=R[i,j]*coefs_qr[j]
    coefs_qr[i]/=R[i,i]

#calculate the predictions using new coefficients
predictions_qr=X@coefs_qr

#calculate the error using the new coefficinets
errors_qr=y-predictions_qr
rel_error_qr=errors_qr/y
print("L2 norm of errors using QR decomposition", np.linalg.norm(errors_qr))
print("L2 norm of relative error using QR decomposition", np.linalg.norm(rel_error_qr))

#save the coefficients to a file
np.savetxt("coefs_qr.csv",coefs_qr,delimiter=",")




###############################   SVD   #############################################
#instead of only 4 features we take all features
X=df.drop(columns=['Price'])
n_samples,n_features=X.shape

#print the number of samples and features
print("number of samples", "number of features:", n_samples,n_features)

X=np.hstack((np.ones((n_samples,1)), X))

#solve the normal equation using SVD
U,S,Vt=np.linalg.svd(X, full_matrices=False)

#X.T@X=V@S.T@S.Vt
#X.T@y=V@S.T@U.T@y
# S.T@S@Vt@coefs=S.T@U.T@y
#b=S^-1@U.T@y

#computing S inverse
S_inv=np.zeros((len(S),len(S)))
for i in range(len(S)):
    S_inv[i,i]=1/S[i]


b=S_inv@U.T@y
coefs_svd=Vt.T@b

#calculate predictions using the new coefficients
predictions_svd=X@coefs_svd


####### Alternate notation ##########
#X.T@X@coefs=X.T@y
#coefs=(X.T@X)^-1@X.T@y
#X_dagger=(X.T@X)^-1@X.T
#find the inverse of X in least square sense
#pseudo inverse of X is X_dagger= V@D^-1@U.T
#so we can compute X_dagger using V, S_inv, U.T
#X_dagger=Vt.T@S_inv@U.T
#coefs_svd=X_dagger@y
#predictions_svd=X@coefs_svd
############### End of alternate notation  ############



#calculate the errors and relative errors
errors_svd=y-predictions_svd
rel_error_svd=errors_svd/y

print("L2 norm of errors using SVD:", np.linalg.norm(errors_svd))
print("L2 norm of relative error using SVD:", np.linalg.norm(rel_error_svd))

#saving the coefs
np.savetxt("coefs_svd.csv", coefs_svd, delimiter=",")



# X_1=X[:,0:1]

# ######### which feature is affecting the most??????


# ### Use X as only square feet to build a linear model to predict price
# X=df["Square_feet"].values
# y=df["Price"].values
# X=np.hstack(np.ones(n_samples,1),X.reshape(-1,1))
# coefs_1=np.linalg.inv(X_1.T@X_1)@X_1.T@y

# predictions_1=X@coefs_1


# #plot the data on X[:,1] vs y axis
# #first make X[:,1] as np.arange between min and max of X[:,1]
# X_feature=np.arange(np.min(X[:,1]),np.max(X[:,1]),0.01)
# plt.scatter(X[:,1],y)
# plt.plot(X_feature,X_feature*coefs_svd[1],color="red")
# #plt.plot(X[:,1],X[:,1:3]@coefs_svd[1:3],color='red')
# plt.xlabel("Square feet")
# plt.ylabel("Price")
# plt.title("Price vs square feet")
# plt.show()
# plt.savefig("Price_vs_square_feet.png")



# plot the data data (X[:,1],y)
# first make X[:,1] as np.arange between min and max of X[:,1]
# then calculate the predictoins using the coefficients
X_features = np.arange(np.min(X[:,1]), np.max(X[:,1]), 0.01)

plt.scatter(X[:,1], y)
plt.plot(X_features,X_features * coefs_svd[1], color='red') #regression line
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.title('Price vs Square Feet')
plt.show()
plt.savefig('Price_vs_Square_Feet.png')






# use x as only saquare feet to build the model

X = df[['Square_Feet']].values
y = df['Price'].values
X= np.hstack([np.ones((n_samples, 1)), X])
 
coeff_1 = np.linalg.inv(X.T @ X) @ X.T @ y
predictions_1 = X @ coeff_1
X_feature = np.arange(np.min(X[:,1]), np.max(X[:,1]), 10)
print("min of X[:,1]", np.min(X[:,1]))
print("max of X[:,1]", np.max(X[:,1]))
# pad the X_feature with ones
X_feature = np.hstack([np.ones((X_feature.shape[0], 1)), X_feature.reshape(-1, 1)])
plt.scatter(X[:,1], y, label='Data')
plt.plot(X_feature[:,1], X_feature  @ coeff_1, label='Regression Line', c='Red')
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.title('Price vs Square Feet')
plt.legend()
plt.show()
plt.savefig('Price_vs_Square_Feet.png')















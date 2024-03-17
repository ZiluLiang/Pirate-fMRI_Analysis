# This script computes (SV)CCA as in https://github.com/google/svcca
acts1 = groundtruth
acts2 = participant

nunsamples = dim(acts1)[1]
numx = dim(acts1)[2]
numy = dim(acts2)[2]

#compute covariance matrix
covariance = cov(t(acts1),t(acts2))
sigmaxx = covariance[1:numx,          1:numx]
sigmaxy = covariance[1:numx,          numx:nunsamples]
sigmayx = covariance[numx:nunsamples, 1:numx]
sigmayy = covariance[numx:nunsamples, numx:nunsamples]

#rescale covariance to make cca computation more stable
xmax = max(abs(sigmaxx))
ymax = max(abs(sigmayy))
sigmaxx = sigmaxx / xmax
sigmayy = sigmayy / ymax
sigmaxy = sigmaxy / sqrt(xmax * ymax)
sigmayx = sigmayx / sqrt(xmax * ymax)

# ([u, s, v], invsqrt_xx, invsqrt_yy,
#   x_idxs, y_idxs) = compute_ccas(sigmaxx, sigmaxy, sigmayx, sigmayy,
#                                  epsilon=epsilon,
#                                  verbose=verbose)

#taking inverse
inv_xx = pinv(sigmaxx)
inv_yy = pinv(sigmayy)

#taking square root
invsqrt_xx = positivedef_matrix_sqrt(inv_xx)
invsqrt_yy = positivedef_matrix_sqrt(inv_yy)

#dot products..
arr = invsqrt_xx %*% (sigmaxy %*% invsqrt_yy)


#"trying to take final svd"
svd_res = svd(arr)
u = svd_res$d
s = svd_res$u
v = svd_res$v

cca = list(coef = s)
positivedef_matrix_sqrt = function(array){
  res = eigen(array)
  w = res$values
  v = res$vectors
  #  A - np.dot(v, np.dot(np.diag(w), v.T))
  wsqrt = sqrt(w)
  sqrtarray = v %*% (diag(wsqrt) %*% t(Conj(v)))
  return(sqrtarray)
}

from ggmfit import *

def ggm_entropy(theta):
    NN = theta.shape[0]
    ent = NN / 2.0 * np.log2(2 * np.pi * np.e) - 1 / 2.0 * np.log2(max(np.linalg.det(theta), 10e-8))
    return ent


def mutual_information(F, Y, G):
    '''
    F is (Sample_Num, Node_Num), stack of feature vectors
    '''
    S = emp_covar_mat(F)
    W, theta, i = ggmfit(S, G, 5000)
    H_X = ggm_entropy(theta)

    unique, counts= np.unique(Y, return_counts=True)
    H_X1Y = 0
    for idx, status in enumerate(unique):
        S = emp_covar_mat(F[Y==status])
        W, theta, i = ggmfit(S, G, 5000)
        H_X1Y += ggm_entropy(theta) * counts[idx]/counts.sum()
    MI_XY = H_X - H_X1Y
    return MI_XY
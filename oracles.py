import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        """
        Computes the value of logistic regression with l2 regularization at point x.
        
        f(x) = 1/m * sum_i log(1 + exp(-b_i * a_i^T x)) + (regcoef / 2) * ||x||^2
        """
        # Compute Ax = A * x
        Ax = self.matvec_Ax(x)
        # Compute z_i = b_i * (a_i^T x) element-wise
        z = self.b * Ax
        # Logistic loss: mean of log(1 + exp(-z))
        # Use np.logaddexp(0, -z) to avoid overflow: log(1 + exp(-z)) = logaddexp(0, -z)
        loss = np.mean(np.logaddexp(0, -z))
        # L2 regularization: (regcoef / 2) * ||x||^2
        reg = 0.5 * self.regcoef * np.dot(x, x)
        return loss + reg

    def grad(self, x):
        """
        Computes the gradient of logistic regression with l2 regularization at point x.
        
        grad f(x) = -1/m * A^T * (b * sigma) + regcoef * x
        where sigma = expit(-b_i * (a_i^T x)) = 1 / (1 + exp(b_i * a_i^T x))
        """
        # Compute Ax = A * x
        Ax = self.matvec_Ax(x)
        # Compute z_i = b_i * (a_i^T x)
        z = self.b * Ax
        # sigma = expit(-z) = 1 / (1 + exp(z)) - numerically stable
        sigma = expit(-z)
        # Gradient of loss: -1/m * A^T * (b * sigma)
        grad_loss = -self.matvec_ATx(self.b * sigma) / len(self.b)
        # Gradient of regularization: regcoef * x
        grad_reg = self.regcoef * x
        return grad_loss + grad_reg

    def hess(self, x):
        """
        Computes the Hessian of logistic regression with l2 regularization at point x.
        
        H(x) = A^T * diag(s) * A + regcoef * I
        where s_i = sigma_i * (1 - sigma_i)
        and sigma_i = expit(-b_i * (a_i^T x))
        """
        # Compute Ax = A * x
        Ax = self.matvec_Ax(x)
        # Compute z_i = b_i * (a_i^T x)
        z = self.b * Ax
        # sigma = expit(-z) = 1 / (1 + exp(z))
        sigma = expit(-z)
        # s = sigma * (1 - sigma) - weights for Hessian
        s = sigma * (1 - sigma)
        # Hessian = A^T * diag(s) * A + regcoef * I
        n = len(x)
        return self.matmat_ATsA(s) + self.regcoef * np.eye(n)


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
        # Cache for Ax values
        self._cached_x = None
        self._cached_Ax = None
        # Cache for directional values
        self._cached_xd = None
        self._cached_d = None
        self._cached_Axd = None
        self._cached_alpha = None
        self._cached_x_alpha = None
        self._cached_Ax_alpha = None

    def _get_Ax(self, x):
        """Helper method to get Ax with caching."""
        if self._cached_x is not None and np.array_equal(self._cached_x, x):
            return self._cached_Ax
        self._cached_x = np.copy(x)
        self._cached_Ax = self.matvec_Ax(x)
        return self._cached_Ax

    def func(self, x):
        Ax = self._get_Ax(x)
        z = self.b * Ax
        loss = np.mean(np.logaddexp(0, -z))
        reg = 0.5 * self.regcoef * np.dot(x, x)
        return loss + reg

    def grad(self, x):
        Ax = self._get_Ax(x)
        z = self.b * Ax
        sigma = expit(-z)
        grad_loss = -self.matvec_ATx(self.b * sigma) / len(self.b)
        grad_reg = self.regcoef * x
        return grad_loss + grad_reg

    def hess(self, x):
        Ax = self._get_Ax(x)
        z = self.b * Ax
        sigma = expit(-z)
        s = sigma * (1 - sigma)
        n = len(x)
        return self.matmat_ATsA(s) + self.regcoef * np.eye(n)

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d) with caching.
        """
        x_alpha = x + alpha * d
        # Check if we have cached value for this point
        if (self._cached_x_alpha is not None and 
            np.array_equal(self._cached_x_alpha, x_alpha)):
            Ax_alpha = self._cached_Ax_alpha
        else:
            Ax_alpha = self.matvec_Ax(x_alpha)
            self._cached_x_alpha = np.copy(x_alpha)
            self._cached_Ax_alpha = Ax_alpha
        
        z = self.b * Ax_alpha
        loss = np.mean(np.logaddexp(0, -z))
        reg = 0.5 * self.regcoef * np.dot(x_alpha, x_alpha)
        return loss + reg

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha} with caching.
        """
        x_alpha = x + alpha * d
        # Check if we have cached value for this point
        if (self._cached_x_alpha is not None and 
            np.array_equal(self._cached_x_alpha, x_alpha)):
            Ax_alpha = self._cached_Ax_alpha
        else:
            Ax_alpha = self.matvec_Ax(x_alpha)
            self._cached_x_alpha = np.copy(x_alpha)
            self._cached_Ax_alpha = Ax_alpha
        
        z = self.b * Ax_alpha
        sigma = expit(-z)
        # phi'(alpha) = gradient at x_alpha dot d
        grad_val = -self.matvec_ATx(self.b * sigma) / len(self.b) + self.regcoef * x_alpha
        return np.squeeze(grad_val.dot(d))


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    # Get dimensions
    m, n = A.shape
    
    def matvec_Ax(x):
        """Computes A * x"""
        return A.dot(x)
    
    def matvec_ATx(x):
        """Computes A^T * x"""
        return A.T.dot(x)
    
    def matmat_ATsA(s):
        """
        Computes A^T * diag(s) * A
        where s is a vector of length m
        """
        # s is a 1D array of length m
        # We need to compute A.T @ diag(s) @ A
        if scipy.sparse.issparse(A):
            # For sparse matrices
            # Multiply A by diag(s) efficiently
            return A.T.dot(A.multiply(s[:, np.newaxis]))
        else:
            # For dense matrices
            # A.T @ (s.reshape(-1, 1) * A)
            return A.T.dot(A * s.reshape(-1, 1))
    
    if oracle_type == 'usual':
        return LogRegL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
    elif oracle_type == 'optimized':
        return LogRegL2OptimizedOracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
    else:
        raise ValueError('Unknown oracle_type=%s' % oracle_type)


def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    n = len(x)
    grad = np.zeros(n)
    f0 = func(x)
    
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = eps
        f_i = func(x + e_i)
        grad[i] = (f_i - f0) / eps
    
    return grad


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    n = len(x)
    hess = np.zeros((n, n))
    f0 = func(x)
    
    # Pre-compute f(x + eps*e_i) for all i
    f_plus = np.zeros(n)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = eps
        f_plus[i] = func(x + e_i)
    
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = eps
        
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = eps
            
            if i == j:
                # Diagonal element: second derivative with respect to i
                # Use 2*eps for central difference formula
                f_2i = func(x + 2 * e_i)
                hess[i, i] = (f_2i - 2 * f_plus[i] + f0) / (eps * eps)
            else:
                # Off-diagonal element
                f_ij = func(x + e_i + e_j)
                hess[i, j] = (f_ij - f_plus[i] - f_plus[j] + f0) / (eps * eps)
    
    return hess
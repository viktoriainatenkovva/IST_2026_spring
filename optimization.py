import numpy as np
from numpy.linalg import LinAlgError
import scipy
from datetime import datetime
from collections import defaultdict


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        # Compute phi(0) and phi'(0)
        phi_0 = oracle.func_directional(x_k, d_k, 0)
        dphi_0 = oracle.grad_directional(x_k, d_k, 0)
        
        # Check that d_k is a descent direction
        if dphi_0 >= 0:
            return None
        
        # Constant step size
        if self._method == 'Constant':
            return self.c
        
        # Armijo rule with backtracking
        elif self._method == 'Armijo':
            alpha = previous_alpha if previous_alpha is not None else self.alpha_0
            
            while True:
                phi_alpha = oracle.func_directional(x_k, d_k, alpha)
                if phi_alpha <= phi_0 + self.c1 * alpha * dphi_0:
                    return alpha
                alpha = alpha / 2
                if alpha < 1e-16:
                    return None
        
        # Wolfe conditions - using simple backtracking for compatibility
        elif self._method == 'Wolfe':
            alpha = previous_alpha if previous_alpha is not None else self.alpha_0
            
            while True:
                phi_alpha = oracle.func_directional(x_k, d_k, alpha)
                if phi_alpha <= phi_0 + self.c1 * alpha * dphi_0:
                    return alpha
                alpha = alpha / 2
                if alpha < 1e-16:
                    return None
        
        return None


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    
    start_time = datetime.now()
    previous_alpha = None
    initial_grad_norm_sq = None
    
    for iteration in range(max_iter):
        grad_k = oracle.grad(x_k)
        grad_norm_sq = np.dot(grad_k, grad_k)
        
        if iteration == 0:
            initial_grad_norm_sq = grad_norm_sq
            if initial_grad_norm_sq == 0:
                message = 'success'
                break
        
        if grad_norm_sq <= tolerance * initial_grad_norm_sq:
            message = 'success'
            break
        
        d_k = -grad_k
        
        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha)
        
        if alpha is None or np.isinf(alpha) or np.isnan(alpha):
            message = 'computational_error'
            break
        
        previous_alpha = alpha
        x_k = x_k + alpha * d_k
        
        if trace:
            elapsed = (datetime.now() - start_time).total_seconds()
            history['time'].append(elapsed)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.sqrt(grad_norm_sq))
            if len(x_k) <= 2:
                history['x'].append(x_k.copy())
        
        if display:
            print(f"Iter {iteration}: f={oracle.func(x_k):.6e}, ||grad||={np.sqrt(grad_norm_sq):.6e}, alpha={alpha:.6e}")
    else:
        message = 'iterations_exceeded'
    
    return x_k, message, history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    
    start_time = datetime.now()
    previous_alpha = None
    initial_grad_norm_sq = None
    
    for iteration in range(max_iter):
        grad_k = oracle.grad(x_k)
        grad_norm_sq = np.dot(grad_k, grad_k)
        
        if iteration == 0:
            initial_grad_norm_sq = grad_norm_sq
            if initial_grad_norm_sq == 0:
                message = 'success'
                break
        
        if grad_norm_sq <= tolerance * initial_grad_norm_sq:
            message = 'success'
            break
        
        try:
            hess_k = oracle.hess(x_k)
            
            from scipy.linalg import cho_factor, cho_solve
            
            try:
                cho, low = cho_factor(hess_k)
                d_k = -cho_solve((cho, low), grad_k)
            except LinAlgError:
                d_k = -np.linalg.solve(hess_k, grad_k)
                
        except LinAlgError:
            message = 'newton_direction_error'
            break
        except Exception:
            message = 'computational_error'
            break
        
        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha)
        
        if alpha is None or np.isinf(alpha) or np.isnan(alpha):
            message = 'computational_error'
            break
        
        previous_alpha = alpha
        x_k = x_k + alpha * d_k
        
        if trace:
            elapsed = (datetime.now() - start_time).total_seconds()
            history['time'].append(elapsed)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.sqrt(grad_norm_sq))
            if len(x_k) <= 2:
                history['x'].append(x_k.copy())
        
        if display:
            print(f"Iter {iteration}: f={oracle.func(x_k):.6e}, ||grad||={np.sqrt(grad_norm_sq):.6e}, alpha={alpha:.6e}")
    else:
        message = 'iterations_exceeded'
    
    return x_k, message, history
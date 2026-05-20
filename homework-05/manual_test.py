import numpy as np
import scipy
import oracles
import optimization

print("Testing imports... OK")

print("\n1. Testing QuadraticOracle...")
A = np.eye(3)
b = np.array([1, 2, 3])
quad = oracles.QuadraticOracle(A, b)
x = np.zeros(3)
assert abs(quad.func(x) - 0.0) < 1e-10, "func failed"
assert np.allclose(quad.grad(x), -b), "grad failed"
print("✓ QuadraticOracle OK")

print("\n2. Testing grad_finite_diff...")
func = lambda x: x[0]**3 + x[1]**2
x = np.array([2.0, 3.0])
g = oracles.grad_finite_diff(func, x, 1e-5)
assert np.allclose(g, [12.0, 6.0], atol=1e-4), "grad_finite_diff failed"
print("✓ grad_finite_diff OK")

print("\n3. Testing hess_finite_diff...")
H = oracles.hess_finite_diff(func, x, 1e-5)
assert np.allclose(H, [[12.0, 0.0], [0.0, 2.0]], atol=1e-3), "hess_finite_diff failed"
print("✓ hess_finite_diff OK")

print("\n4. Testing LogRegL2Oracle...")
A = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
b = np.array([1, 1, -1, 1])
reg_coef = 0.5
logreg = oracles.create_log_reg_oracle(A, b, reg_coef, 'usual')
x = np.zeros(2)
f = logreg.func(x)
assert abs(f - 0.693147180) < 1e-6, f"func failed: got {f}"
g = logreg.grad(x)
assert np.allclose(g, [0, -0.25], atol=1e-6), f"grad failed: got {g}"
print("✓ LogRegL2Oracle OK")

print("\n5. Testing gradient_descent...")
A = np.eye(3)
b = np.array([1, 2, 3])
quad = oracles.QuadraticOracle(A, b)
x0 = np.ones(3) * 10.0
x_star, msg, _ = optimization.gradient_descent(quad, x0, max_iter=1, tolerance=1e-5)
assert np.allclose(x_star, [1.0, 2.0, 3.0], atol=1e-5), f"gradient_descent failed: got {x_star}"
print("✓ gradient_descent OK")

print("\n6. Testing LineSearchTool...")
ls_tool = optimization.LineSearchTool(method='Constant', c=1.0)
x = np.array([100, 0, 0])
d = np.array([-1, 0, 0])
alpha = ls_tool.line_search(quad, x, d)
assert alpha == 1.0, f"line_search failed: got {alpha}"
print("✓ LineSearchTool OK")

print("\n" + "="*50)
print("ALL TESTS PASSED!")

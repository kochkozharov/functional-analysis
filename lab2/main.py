import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

a = 0.0
b = 1.6

def weight(t):
    return (6-t) * t

def dot_product(p, q):
    integrand = lambda t: p(t) * q(t) * weight(t)
    return quad(integrand, a, b)[0]

def norm(p):
    return np.sqrt(dot_product(p, p))

def metric(p, q):
    diff = lambda t: p(t) - q(t)
    return np.sqrt(dot_product(diff, diff))

def y_func(t):
    return np.exp(t)

def create_next_orthonormal_element(term, basis):
    p = np.poly1d([1] + [0]*term)
    for q in basis:
        proj = dot_product(p, q)
        p = p - proj * q
    norm_val = norm(p)
    return p / norm_val

def compute_fourier_coefficient(q_norm):
    return dot_product(y_func, q_norm)

def incremental_approximation(epsilon, max_terms=50):
    basis = []
    coeffs = []
    approximations = []
    errors = []

    current_approx = np.poly1d([0])
    error = metric(y_func, current_approx)
    errors.append(error)
    print(f"Начальная ошибка: {error:.3e}")

    term = 0
    while error > epsilon and term < max_terms:
        q_norm = create_next_orthonormal_element(term, basis)
        term += 1
        if q_norm is None:
            continue
        basis.append(q_norm)
        c = compute_fourier_coefficient(q_norm)
        coeffs.append(c)
        current_approx = current_approx + c * q_norm
        error = metric(y_func, current_approx)
        errors.append(error)
        approximations.append(current_approx)
        print(f"Добавлен элемент {len(basis)}: степень {term-1}, ошибка = {error:.3e}")
    return basis, coeffs, approximations, errors

epsilon = 1e-3
basis, coeffs, approximations, errors = incremental_approximation(epsilon)

t_vals = np.linspace(a, b, 300)
y_vals = np.array([y_func(t) for t in t_vals])

final_N = len(basis)
final_approx_vals = approximations[-1](t_vals)
print(f"\nФинальное число членов ряда: {final_N}")

plt.figure(figsize=(14, 10))
for idx, approx_poly in enumerate(approximations):
    approx_vals = approx_poly(t_vals)
    plt.plot(t_vals, approx_vals, label=f"N={idx+1}, error={errors[idx+1]:.1e}")
plt.plot(t_vals, y_vals, 'k-', lw=3, label="$y(t)=e^t$")
plt.xlabel("t")
plt.ylabel("$y(t)$")
plt.title("Промежуточные приближения функции $y(t)=e^t$")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

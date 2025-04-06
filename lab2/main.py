import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import math

# Параметры задачи
a = 0.0
b = 1.6  # [0, 0.8 + 8/10]
def weight(t):
    return 6 * t**2

# Внутреннее произведение для полиномов: <p, q> = ∫_a^b p(t)q(t) weight(t) dt
def dot_product(p, q):
    f = lambda t: p(t) * q(t) * weight(t)
    return quad(f, a, b)[0]

def norm(p):
    return np.sqrt(dot_product(p, p))

def metric(p, q):
    diff = lambda t: p(t) - q(t)
    return np.sqrt(dot_product(diff, diff))

# Процесс Грама–Шмидта для мономов с представлением через numpy.poly1d
def gram_schmidt_polynomials(n_max):
    # Строим ортонормированный базис из мономов t^0, t^1, ..., t^(n_max-1)
    orthonormal_basis = []
    for n in range(n_max):
        # Моном: t^n
        # Для n=0: [1], для n=1: [1, 0] (что соответствует t), для n=2: [1, 0, 0] (t^2) и т.д.
        p = np.poly1d([1] + [0]*n)
        # Ортогонализация: вычитаем проекции на уже найденные элементы базиса
        for q in orthonormal_basis:
            proj = dot_product(p, q)
            p = p - proj * q
        norm_val = norm(p)
        # if norm_val < 1e-12:
        #     continue
        q_norm = p / norm_val
        orthonormal_basis.append(q_norm)
    return orthonormal_basis

# Заданная функция y(t)
def y_func(t):
    return np.exp(t)

# Вычисление коэффициентов Фурье: c_k = <y, phi_k>
def compute_fourier_coefficients(basis):
    coeffs = []
    for q in basis:
        c = dot_product(y_func, q)
        coeffs.append(c)
    return coeffs

# Частичная сумма ряда Фурье S_N(t) = sum_{k=0}^{N-1} c_k * phi_k(t)
def fourier_partial_sum(t, basis, coeffs, N):
    s = 0
    for k in range(N):
        s += coeffs[k] * basis[k](t)
    return s

# Вычисление среднеквадратичной ошибки аппроксимации
def compute_error(basis, coeffs, N):
    approx = lambda t: fourier_partial_sum(t, basis, coeffs, N)
    err_sq = metric(y_func, approx)
    # err_sq = quad(lambda t: (y_func(t) - approx(t))**2 * weight(t), a, b)[0]
    return err_sq

# Настройки вычислений
n_max = 20  # можно увеличить при наличии ресурсов
basis = gram_schmidt_polynomials(n_max)
coeffs = compute_fourier_coefficients(basis)

# Определяем минимальное число членов ряда для достижения заданной точности
epsilons = [1e-1, 1e-2, 1e-3]
N_required = {}  # для каждого eps минимальное число членов N
for eps in epsilons:
    for N in range(1, len(basis)+1):
        err = compute_error(basis, coeffs, N)
        if err < eps:
            N_required[eps] = N
            print(f"Для ε = {eps:.1e} достигнута ошибка E = {err:.3e} при N = {N}")
            break
    else:
        print(f"Для ε = {eps:.1e} не достигнута требуемая точность с N = {len(basis)}")

# Зададим сетку для построения графиков
t_vals = np.linspace(a, b, 300)
y_vals = np.array([y_func(t) for t in t_vals])

# График аппроксимации функции y(t) = exp(t)
plt.figure(figsize=(12, 8))
plt.plot(t_vals, y_vals, 'k-', lw=2, label="$y(t)=e^t$")
colors = ['r', 'b', 'g']
for idx, eps in enumerate(epsilons):
    N = N_required.get(eps, len(basis))
    approx_vals = np.array([fourier_partial_sum(t, basis, coeffs, N) for t in t_vals])
    plt.plot(t_vals, approx_vals, color=colors[idx], linestyle='--',
             label=f"Частичная сумма N={N} (ε={eps:.0e})")

plt.xlabel("t")
plt.ylabel("$y(t)=e^t$")
plt.title("Аппроксимация функции $y(t)=e^t$ рядом Фурье в $L^2_f([0,1.6])$")
plt.legend()
plt.grid(True)
plt.show()

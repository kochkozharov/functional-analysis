import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import math

# Параметры задачи
a = 0.0
b = 1.6  # [0, 0.8 + 8/10]

def weight(t):
    return 6 * t**2

# Внутреннее произведение: <p, q> = ∫_a^b p(t)q(t)*weight(t) dt
def dot_product(p, q):
    integrand = lambda t: p(t) * q(t) * weight(t)
    return quad(integrand, a, b)[0]

def norm(p):
    return np.sqrt(dot_product(p, p))

# Метрика: d(p,q)=||p-q|| в L^2_f
def metric(p, q):
    diff = lambda t: p(t) - q(t)
    return np.sqrt(dot_product(diff, diff))

# Заданная функция y(t)
def y_func(t):
    return np.exp(t)

# Функция для создания следующего ортогонального (нормированного) элемента
def create_next_orthonormal_element(term, basis):
    p = np.poly1d([1] + [0]*term)  # Моном t^term
    for q in basis:
        proj = dot_product(p, q)
        p = p - proj * q
    norm_val = norm(p)
    # if norm_val < 1e-12:
    #     return None
    return p / norm_val

# Функция для вычисления коэффициента Фурье для данного базисного элемента
def compute_fourier_coefficient(q_norm):
    return dot_product(y_func, q_norm)

# Функция, которая постепенно строит приближение в виде np.poly1d
def incremental_approximation(epsilon, max_terms=50):
    basis = []          # список ортонормированных базисных функций (np.poly1d)
    coeffs = []         # коэффициенты Фурье для y(t)
    approximations = [] # список аппроксимаций в виде np.poly1d
    errors = []         # список ошибок аппроксимации после каждого шага

    current_approx = np.poly1d([0])  # Начальное приближение: нулевой полином
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
        # Обновляем полином-аппроксимацию: просто прибавляем c * q_norm
        current_approx = current_approx + c * q_norm
        error = metric(y_func, current_approx)
        errors.append(error)
        approximations.append(current_approx)
        print(f"Добавлен элемент {len(basis)}: степень {term-1}, ошибка = {error:.3e}")
    return basis, coeffs, approximations, errors

# Задаём желаемую точность
epsilon = 1e-3
basis, coeffs, approximations, errors = incremental_approximation(epsilon, max_terms=50)

# Определяем сетку для построения графиков
t_vals = np.linspace(a, b, 300)
y_vals = np.array([y_func(t) for t in t_vals])

# Финальное приближение
final_N = len(basis)
final_approx_vals = approximations[-1](t_vals)
print(f"\nФинальное число членов ряда: {final_N}")

# График исходной функции и финального приближения
plt.figure(figsize=(12, 8))
plt.plot(t_vals, y_vals, 'k-', lw=2, label="$y(t)=e^t$")
plt.plot(t_vals, final_approx_vals, 'r--', lw=2, label=f"Частичная сумма N={final_N} (ε={epsilon:.0e})")
plt.xlabel("t")
plt.ylabel("$y(t)$")
plt.title("Аппроксимация функции $y(t)=e^t$ рядом Фурье в $L^2_f([0,1.6])$")
plt.legend()
plt.grid(True)
plt.show()

# Демонстрация промежуточных приближений
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

import inspect
import numpy as np
from numpy import exp, sin, cos
import matplotlib.pyplot as plt

def T(x):
    def new_x(t):
        if t >= 0 and t <= 1/3:
            return (1/9)*x(3*t)-15/2
        if t > 1/3 and t < 2/3:
            return (1/3)*(x(0)-x(1)+135)*t+(2/9)*x(1)-(1/9)*x(0)-135/6
        if t >= 2/3 and t <= 1:
            return (1/9)*x(3*t-2)+15/2
        raise ValueError("t must be in [0,1]")
    return new_x

def n_iters(alpha, eps, initial_rho):
    return np.log(eps*(1-alpha)/initial_rho)/np.log(alpha)

def C01_metric(f, g, n_samples=1000):
    x = np.linspace(0, 1, n_samples)
    f_vec = np.vectorize(f)
    g_vec = np.vectorize(g)
    return np.max(np.abs(f_vec(x) - g_vec(x)))

def main():
    eps=1e-3
    alpha=1/9
    f = lambda t: sin(t)
    lambda_source = inspect.getsource(f)
    lambda_body = lambda_source.split(":", 1)[1].strip()
    f_1 = T(f)
    initial_rho = C01_metric(f, f_1)
    iters = int(np.ceil(n_iters(alpha, eps, initial_rho)))
    for i in range(0, iters):
        f = T(f)
    f = np.vectorize(f)
    x = np.linspace(0, 1, 10000)

    y = f(x)

    plt.scatter(x, y, s=1)
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.grid(True)
    plt.suptitle(f'График неподвижной точки с точностью {eps}')
    plt.figtext(0.5, 0.01, f'Начальное приближение: {lambda_body}, число итераций: {iters}', ha='center')
    plt.show()
    return

if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

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

def iters_for_eps(alpha, eps, initial_rho):
    return np.log(eps*(1-alpha)/initial_rho)/np.log(alpha)

def C01_metric(f1, f2):
    return minimize_scalar(lambda x: -abs(f1(x) - f2(x)), bounds=(0, 1), method='bounded').x

def main():
    eps=10e-7
    alpha=1/9
    f = lambda x: 100000*np.exp(x)
    f_1 = T(f)
    initial_rho = C01_metric(f, f_1)
    iters = int(np.ceil(iters_for_eps(alpha, eps, initial_rho)))
    print(f"Iterations for eps={eps}: {iters}")
    for i in range(0, iters):
        f = T(f)
    f = np.vectorize(f)
    x = np.linspace(0, 1, 1000)

    y = f(x)

    plt.scatter(x, y, s=1)
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.grid(True)
    plt.show()
    return

if __name__ == "__main__":
    main()
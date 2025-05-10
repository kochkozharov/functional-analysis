from scipy.integrate import quad
import numpy as np

k = 8
l = 15

def H(x):
    return np.heaviside(x, 1)

def f(x):
    return np.sin(k*x)-3*H(2*x-l/5)+3*x**2

def F_cont(x):
    return np.exp(x)+x**3

def F_disc(x):
    return H(x-1)+2*H(x-k)

def F(x):
    return F_disc(x) + F_cont(x)

def f_cont(x):
    return f(x)*(np.exp(x) + 3*x**2)

I_part1, err = quad(f_cont, -8, 75)
I_part2=f(1)+2*f(8)
I = I_part1 + I_part2
print("Интеграл Лебега-Стилтьеса:", I)
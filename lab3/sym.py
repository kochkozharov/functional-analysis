import sympy as sp
from scipy.integrate import quad

k = 8
l = 15

x = sp.symbols('x')

f = sp.sin(k*x)-3*sp.Heaviside(2*x-l/5, 1)+3*x**2

F_cont = sp.exp(x)+x**3

F_disc = sp.Heaviside(x-1, 1)+2*sp.Heaviside(x-k, 1)

F = F_disc + F_cont

f_cont = f*sp.diff(F_cont, x)

I_part1_sym = sp.integrate(f_cont, (x, -8, 75))
I_part2_sym = f.subs(x, 1)+2*f.subs(x, 8)
I_sym = sp.simplify(I_part1_sym + I_part2_sym, rational=True)
print("Аналитическое выражение интеграла Лебега-Стилтьеса:\n", I_sym)
latex_str = sp.latex(I_sym)
print("Latex выражение:\n", latex_str)
I_sym_val = I_sym.evalf()
print("Численное значение интеграла Лебега-Стилтьеса в результате вычисления символьного выражения", I_sym_val)

f_cont_num = sp.lambdify(x, f_cont, modules=['numpy'])
f_num = sp.lambdify(x, f, modules=['numpy'])

I_part1, err = quad(f_cont_num, -8, 75)
I_part2=f_num(1)+2*f_num(8)
I = I_part1 + I_part2
print("Численное значение интеграла Лебега-Стилтьеса в результате численного интегрирования:", I)
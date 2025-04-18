import numpy as np
import matplotlib.pyplot as plt

def rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h/2, y + h/2 * k1)
    k3 = f(t + h/2, y + h/2 * k2)
    k4 = f(t + h, y + h * k3)
    return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

def integrate_rk4(f, t0, T, y0, h):
    n_steps = int((T - t0)/h) + 1
    t_values = np.linspace(t0, T, n_steps)
    y_values = np.zeros((n_steps, len(y0)))
    y_values[0] = y0
    for i in range(n_steps - 1):
        y_values[i+1] = rk4_step(f, t_values[i], y_values[i], h)
    return t_values, y_values

N = 3

a11 = 0.01 * N
a12 = 0.0001 * N
a21 = 0.0001 * N
a22 = 0.04 * N

def lotka_volterra(t, vars):
    x, y = vars
    dxdt = a11 * x - a12 * x * y
    dydt = a21 * x * y - a22 * y
    return np.array([dxdt, dydt])

x0 = 1000 - 10 * N
y0 = 700 - 10 * N
y0_task1 = np.array([x0, y0])

t0 = 0
T1 = 300
h = 0.1

t_task1, sol_task1 = integrate_rk4(lotka_volterra, t0, T1, y0_task1, h)

plt.figure()
plt.plot(t_task1, sol_task1[:, 0], label='Жертви (x)')
plt.plot(t_task1, sol_task1[:, 1], label='Хижаки (y)')
plt.xlabel('Час, дні')
plt.ylabel('Кількість')
plt.title('Залежність кількості жертв та хижаків від часу')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(sol_task1[:, 0], sol_task1[:, 1])
plt.xlabel('Кількість жертв (x)')
plt.ylabel('Кількість хижаків (y)')
plt.title('Фазовий портрет системи Лотки-Вольтери')
plt.grid(True)
plt.show()

delta = 0.5

H = 1000 - N
beta = 25 - N
gamma = 1 / N

x0_epi = 900 - N
y0_epi = 90 - N
z0_epi = H - x0_epi - y0_epi
y0_task2 = np.array([x0_epi, y0_epi, z0_epi])
w0_epi = 0

def epidemic_model(t, vars):
    x, y, z = vars
    dxdt = - beta * x * y / H
    dydt = beta * x * y / H - gamma * y - delta * y
    dzdt = gamma * y
    return np.array([dxdt, dydt, dzdt])

T2 = 40

t_task2, sol_task2 = integrate_rk4(epidemic_model, t0, T2, y0_task2, h)

plt.figure()
plt.plot(t_task2, sol_task2[:, 0], label='Здорові (x)')
plt.plot(t_task2, sol_task2[:, 1], label='Хворі (y)')
plt.plot(t_task2, sol_task2[:, 2], label='Одужали (z)')
plt.xlabel('Час, дні')
plt.ylabel('Кількість')
plt.title('Залежність кількості здорових, хворих та одужалих від часу')
plt.legend()
plt.grid(True)
plt.show()

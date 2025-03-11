import numpy as np
import plotly.graph_objects as go

a0, a1, a2 = 0, 0, 0

def gauss_jordan(matrix, size):
    for i in range(size):
        diag = matrix[i][i]
        if diag == 0:
            print("Система не має розвʼязку або має безліч розвʼязків.")
            return

        for k in range(size + 1):
            matrix[i][k] /= diag

        for j in range(size):
            if j != i:
                factor = matrix[j][i]
                for k in range(size + 1):
                    matrix[j][k] -= factor * matrix[i][k]

def func(x1, x2):
    return a0 + a1 * x1 + a2 * x2

def r_squared(data):
    n = len(data[0])
    numerator = 0
    denominator = 0
    y_mean = np.mean(data[2])

    for i in range(n):
        x1 = data[0][i]
        x2 = data[1][i]
        y = data[2][i]
        numerator += (func(x1, x2) - y) ** 2
        denominator += (y - y_mean) ** 2

    return 1 - numerator / denominator if denominator != 0 else float('nan')

def plot_data_and_equation(data):
    scatter = go.Scatter3d(
        x=data[0], y=data[1], z=data[2], mode='markers',
        marker=dict(size=5, color='gray', opacity=0.8), name='Дані'
    )

    x1_vals = np.linspace(-1, 3, 100)
    x2_vals = np.linspace(-1, 3, 100)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Y_vals = a0 + a1 * X1 + a2 * X2

    surface = go.Surface(
        x=X1, y=X2, z=Y_vals, opacity=0.3, colorscale='blues', name='МНК Рівняння'
    )

    x1_solution = 1.5
    x2_solution = 3
    y_solution = a0 + a1 * x1_solution + a2 * x2_solution
    solution_point = go.Scatter3d(
        x=[x1_solution], y=[x2_solution], z=[y_solution],
        mode='markers', marker=dict(size=10, color='red'), name='Точка рішення'
    )

    fig = go.Figure(data=[scatter, surface, solution_point])
    fig.update_layout(
        scene=dict(xaxis_title='X1', yaxis_title='X2', zaxis_title='Y'),
        title='Метод найменших квадратів',
        showlegend=True
    )
    fig.show()

def main():
    data = np.array([
        [0, 0, 0, 1, 1, 2, 2, 2],
        [1.5, 2.5, 3.5, 1.5, 3.5, 1.5, 2.5, 2.5],
        [2.3, 4.9, 1.7, 4.4, 3.4, 6.7, 6.2, 7.2]
    ])

    n = len(data[0])
    sum_x1, sum_x2, sum_y = sum(data[0]), sum(data[1]), sum(data[2])
    sum_x1x1 = sum(x ** 2 for x in data[0])
    sum_x1x2 = sum(data[0][i] * data[1][i] for i in range(n))
    sum_x2x2 = sum(x ** 2 for x in data[1])
    sum_x1y = sum(data[0][i] * data[2][i] for i in range(n))
    sum_x2y = sum(data[1][i] * data[2][i] for i in range(n))

    matrix = [
        [n, sum_x1, sum_x2, sum_y],
        [sum_x1, sum_x1x1, sum_x1x2, sum_x1y],
        [sum_x2, sum_x1x2, sum_x2x2, sum_x2y]
    ]

    gauss_jordan(matrix, 3)

    global a0, a1, a2
    a0, a1, a2 = matrix[0][3], matrix[1][3], matrix[2][3]

    print("Розвʼязання системи рівнянь:")
    print(f"a0 = {a0}")
    print(f"a1 = {a1}")
    print(f"a2 = {a2}")
    print(f"\nЗначення функції у точці (1.5, 3): {func(1.5, 3)}")
    print(f"\nКоефіцієнт детермінації R²: {r_squared(data)}")

    plot_data_and_equation(data)

if __name__ == "__main__":
    main()
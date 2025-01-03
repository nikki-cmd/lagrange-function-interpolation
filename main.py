import numpy as np

xi = np.array([-1.02, -1.2, -1.5, -1.79, -1.98]) 
yi = np.array([0.34, 0.177, 0.03, 0.003, 0.0004]) 


def get_coefficients(_pl: int, _xi: np.ndarray):
    n = int(_xi.shape[0])
    coefficients = np.empty((n, 2), dtype=float)
    for i in range(n):
        if i == _pl:
            coefficients[i][0] = float('inf')
            coefficients[i][1] = float('inf')
        else:
            coefficients[i][0] = 1 / (_xi[_pl] - _xi[i])
            coefficients[i][1] = -_xi[i] / (_xi[_pl] - _xi[i])
    filtered_coefficients = np.empty((n - 1, 2), dtype=float)
    j = 0
    for i in range(n):
        if coefficients[i][0] != float('inf'):
            filtered_coefficients[j][0] = coefficients[i][1]
            filtered_coefficients[j][1] = coefficients[i][0]
            j += 1
    return filtered_coefficients


def get_polynomial_l(_xi: np.ndarray):
    n = int(_xi.shape[0])
    pli = np.zeros((n, n), dtype=float)
    for pl in range(n):
        coefficients = get_coefficients(pl, _xi)
        for i in range(1, n - 1): 
            if i == 1: 
                pli[pl][0] = coefficients[i - 1][0] * coefficients[i][0]
                pli[pl][1] = coefficients[i - 1][1] * coefficients[i][0] + coefficients[i][1] * coefficients[i - 1][0]
                pli[pl][2] = coefficients[i - 1][1] * coefficients[i][1]
            else:
                clone_pli = np.zeros(n, dtype=float)
                for val in range(n):
                    clone_pli[val] = pli[pl][val]
                zeros_pli = np.zeros(n, dtype=float)
                for j in range(n-1):  
                    product_1 = clone_pli[j] * coefficients[i][0]
                    product_2 = clone_pli[j] * coefficients[i][1]
                    zeros_pli[j] += product_1
                    zeros_pli[j+1] += product_2
                for val in range(n):
                    pli[pl][val] = zeros_pli[val]
    return pli

def get_polynomial(_xi: np.ndarray, _yi: np.ndarray):
    n = int(_xi.shape[0])
    polynomial_l = get_polynomial_l(_xi)
    for i in range(n):
        for j in range(n):
            polynomial_l[i][j] *= _yi[i]
    L = np.zeros(n, dtype=float)
    for i in range(n):
        for j in range(n):
            L[i] += polynomial_l[j][i]
    return L
print(get_polynomial(xi, yi))
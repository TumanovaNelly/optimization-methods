import numpy as np

z = np.array(
    [1, 2, 4, 1, 0, 0, 0, 0]
)  # --> max
z = z.astype(np.float64)

# расширенная матрица ограничений в каноническом виде с выделенным базисом
f_expanded = np.array([
    [1, 1, 1, 0, 1, 0, 0, 0,   10],
    [0, 1, 2, 1, 0, 0, 1, 0,   6],
    [1, 0, 0, 1, 0, -1, 0, 1,  2]
])
f_expanded = f_expanded.astype(np.float64)

# индексы базисных векторов
base_indexes = [4, 6, 7]

# текущий максимум z
z_mx = -np.inf

while True:
    # пересчитываем нижнюю строку
    d = np.array([-(f_expanded[:, i] @ z[base_indexes]) + z[i] for i in range(len(z))])
    z_mx = f_expanded[:, -1] @ z[base_indexes]

    # находим разрешающий элемент
    col_index = np.argmax(d)
    if d[col_index] <= 0.0: break
    row_index = 0
    cur_min = np.inf
    for i in range(len(f_expanded)):
        if f_expanded[i][col_index] <= 0.0: continue
        val = f_expanded[i][-1] / f_expanded[i][col_index]
        if cur_min > val:
            cur_min = val
            row_index = i

    # меняем базис
    base_indexes[row_index] = col_index
    f_expanded[row_index] /= f_expanded[row_index][col_index]
    for i in range(len(f_expanded)):
        if i == row_index: continue
        f_expanded[i] -= f_expanded[row_index] * f_expanded[i][col_index]

answer = np.zeros_like(z)
answer[base_indexes] = f_expanded[:, -1]
print(answer)

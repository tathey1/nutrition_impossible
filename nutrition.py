from scipy.optimize import least_squares
from scipy.linalg import svd
import numpy as np
#fat, cholesterol, sodium, carbs, fiber, protein, vitamin A, vitamin B12, Vitamin C, Vitamin D, Calcium, Iron, Potassium

nutrients = ['fat', 'cholesterol', 'sodium', 'carbs', 'fiber', 'protein',
'vitamin A', 'vitamin B12', 'Vitamin C', 'Vitamin D', 'Calcium', 'Iron', 'Potassium']
foods = ['egg','wheat bread','lettuce','raw chicken','carrots','apple','dates','pb','salmon',
'sheeps milk','orange','bananas','mushrooms','quaker oatmeal squares','potato','brussel sprouts','avocado']

A = np.array([[12, 124, 6, 0, 0, 26, 0, 37, 0, 10, 4, 10, 3], #egg
    [4, 0, 26, 17, 21, 17, 0, 0, 0, 0, 6, 16, 4], #wheat bread
    [0, 0, 1, 1, 5, 3, 148, 0, 10, 0, 3, 5, 4], #lettuce
    [10, 29, 3, 0, 0, 34, 0, 23, 0, 0, 0, 5, 11], #raw chicken ground
    [0, 0, 3, 3, 10, 2, 334, 0, 7, 0, 3, 2, 7], #raw carrots
    [0, 0, 0, 5, 9, 1, 1, 0, 5, 0, 0, 1, 2], #apple
    [0, 0, 0, 27, 24, 4, 3, 0, 3, 0, 5, 5, 15], #dates
    [64, 0, 21, 9, 20, 44, 0, 0, 0, 0, 4, 12, 13], #peanut butter
    [6, 15, 3, 0, 0, 42, 2, 173, 0, 54, 1, 2, 8], #raw pink salmon
    [9, 9, 2, 2, 0, 12, 3, 30, 5, 0, 15, 1, 3], #sheeps milk
    [0, 0, 0, 9, 39, 3, 8, 0, 151, 0, 12, 4, 5], #oranges
    [0, 0, 0, 8, 9, 2, 1, 0, 10, 0, 0, 1, 8], #bananas
    [0, 0, 0, 1, 4, 6, 0, 2, 2, 1, 0, 3, 7], #mushrooms white
    [6, 0, 15, 28, 30, 22, 24, 0, 13, 0, 15, 163, 8], #quaker oatmeal squares
    [0, 0, 0, 4, 9, 5, 0, 0, 13, 0, 2, 18, 9], #potato with skin
    [0, 0, 1, 3, 14, 7, 15, 0, 94, 0, 3, 8, 8], #brussel sprout
    [13, 0, 0, 3, 20, 4, 3, 0, 19, 0, 1, 1, 7], #avocado
    ]).T


def sq_diff(c):
    n = A.shape[0]
    hundreds = np.ones(n)*100
    residual = A@c - hundreds
    return residual

x0 = np.ones(A.shape[1])
sol = least_squares(sq_diff, x0, bounds=(0,200))

print(sol['cost'])
print(sol['success'])
print(sol['fun'])
print(sol['x'])
print(A@sol['x'])
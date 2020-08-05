from scipy.optimize import least_squares
import numpy as np 

foods = np.load("C:\\Users\\Thomas Athey\\Documents\\Other\\nutrition_impossible\\data\\foods.npy")
num_foods = len(foods)
nutrients = np.load("C:\\Users\\Thomas Athey\\Documents\\Other\\nutrition_impossible\\data\\nutrients.npy")
num_nutrients = len(nutrients)
pcts = np.load("C:\\Users\\Thomas Athey\\Documents\\Other\\nutrition_impossible\\data\\percent_daily_value.npy")
pcts = pcts.astype(float)

def sq_diff(c):
    hundreds = np.ones(num_nutrients)*100
    residual = pcts.T@c - hundreds
    return residual

x0 = np.ones(num_foods)
sol = least_squares(sq_diff, x0, bounds=(0,200), verbose=2)

print(sol['cost'])
print(sol['success'])
print(sol['fun'])
print(sol['x'])
print(pcts.T@c['x'])
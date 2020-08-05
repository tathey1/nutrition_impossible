from scipy.optimize import least_squares
import numpy as np 
import random
from sklearn.linear_model import Lasso
import pandas as pd

base1 = "C:\\Users\\Thomas Athey\\Documents\\Other\\nutrition_impossible\\data\\"
base2 = "/cis/home/tathey/projects/other/nutrition_impossible/data/"
base = base1
num_foods = 8790
option = 2 #1 least squares 2 lasso


foods = np.load(base + "foods.npy")
num_foods_total = len(foods)
print(num_foods_total)
sample = random.sample(range(num_foods_total),num_foods)
foods = foods[sample]

nutrients = np.load(base + "nutrients.npy")
print(nutrients)
num_nutrients = len(nutrients)

pcts = np.load(base + "percent_daily_value.npy")
pcts = pcts[sample,:]
pcts = pcts.astype(float)

def sq_diff(c):
    hundreds = np.ones(num_nutrients)*100
    residual = pcts.T@c - hundreds
    return residual

if option == 1:
    x0 = np.ones(num_foods)
    sol = least_squares(sq_diff, x0, bounds=(0,200), verbose=2)


    print(sol['cost'])
    print(sol['success'])
    print(sol['fun'])
    print(np.mean(np.abs(sol['fun'])))
    amts = dict(zip(foods,sol['x']))
    print(amts)
elif option ==2:
    clf = Lasso(alpha=9000,fit_intercept=False,positive=True)
    clf.fit(pcts.T,np.ones(num_nutrients)*100)
    c = clf.coef_
    resid = sq_diff(c)
    print(resid)
    print(np.mean(np.abs(resid)))
    idxs = c>0.01
    c = c[idxs]
    foods = foods[idxs]
    amts = dict(zip(foods,c*100))
    print(amts)
    print(len(foods))

    d = {'food':foods, 'amount (g)':c*100}
    df = pd.DataFrame(data=d)
    df.to_csv("C:\\Users\\Thomas Athey\\Documents\\Other\\nutrition_impossible\\data\\results.csv")



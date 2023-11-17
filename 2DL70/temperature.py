import pandas as pd
import numpy as np


T = pd.read_csv('../data/Eindhoven 2022.csv', index_col=0)

Tmax = T['Max Temperature'].to_numpy()
Tmin = T['Min Temperature'].to_numpy()
assert np.all(Tmax > Tmin)

div = 5
Tmax = div*(Tmax/div).astype(int)
Tmin = div*(Tmin/div).astype(int)
ITmax = {value : key for (key, value) in enumerate(np.unique(Tmax))}
ITmin = {value : key for (key, value) in enumerate(np.unique(Tmin))}
T = np.zeros((len(ITmax), len(ITmin)))
for x, y in  zip(Tmax, Tmin):
	T[ITmax[x], ITmin[y]] += 1

print(ITmax)
print(ITmin)
print(T)
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit as sss
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

housedata = pd.read_csv("data.csv")

split = sss(n_splits=1, test_size= 0.2, random_state=42)
for trindex, teindex in split.split(housedata, housedata['CHAS']):
    trainset = housedata.loc[trindex]
    testset = housedata.loc[teindex]

housing_lables = trainset["MEDV"].copy()
t_lables = testset["MEDV"].copy()
trainset = trainset.drop("MEDV", axis = 1)
testset = testset.drop("MEDV", axis = 1)
housedata = housedata.drop("MEDV", axis = 1)



pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy= "median")),
    ('std_scaler', StandardScaler()),
])

tr_num_housing = pipeline.fit_transform(trainset)


model = RandomForestRegressor()
model.fit(tr_num_housing, housing_lables)

sdata = trainset.iloc[:5]
slables = housing_lables.iloc[:5]
pdata = pipeline.transform(sdata)
arr = model.predict(pdata)
print(arr)
print(list(slables))

tdata = pipeline.transform(testset)
p = model.predict(tdata)
print(p)
print(t_lables)

mse = mean_squared_error(t_lables, p)
rmse = np.sqrt(mse)
print(rmse)
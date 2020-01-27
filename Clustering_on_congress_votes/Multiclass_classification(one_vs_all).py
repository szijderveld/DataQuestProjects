import pandas as pd
import numpy as np
cars = pd.read_csv("auto.csv")
print(cars.head())
unique_regions = cars["origin"].unique()
print(unique_regions)




#use dummy variables to split cars by number of cyliners and year (use binary classification)
dummy_cylinders = pd.get_dummies(cars["cylinders"], prefix="cyl")
cars = pd.concat([cars, dummy_cylinders], axis=1)
dummy_years = pd.get_dummies(cars["year"], prefix="year")
cars = pd.concat([cars, dummy_years], axis=1)
cars = cars.drop("year", axis=1)
cars = cars.drop("cylinders", axis=1)
print(cars.head())



#randomise ordering or rows and split data into training and testing sets
shuffled_rows = np.random.permutation(cars.index)
shuffled_cars = cars.iloc[shuffled_rows]
highest_train_row = int(cars.shape[0] * .70)
train = shuffled_cars.iloc[0:highest_train_row]
test = shuffled_cars.iloc[highest_train_row:]



# using the one vs all method, choosing a single category as the positive and the rest as the negative. Repeat 3 times in our case to cover all bases and pick largest probability
from sklearn.linear_model import LogisticRegression

unique_origins = cars["origin"].unique()
unique_origins.sort()


models = {}		#to be filled by the three models
features = [c for c in train.columns if c.startswith("cyl") or c.startswith("year")]		#select features that refer to cylinder and year

for origin in unique_origins:
    model = LogisticRegression()
    
    X_train = train[features]
    y_train = train["origin"] == origin

    model.fit(X_train, y_train)
    models[origin] = model   #fill dictionary with the trained model for 'NA' as true, 'Europe' and 'Asia'



testing_probs = pd.DataFrame(columns=unique_origins)
testing_probs = pd.DataFrame(columns=unique_origins)  

for origin in unique_origins:
    # Select testing features.
    X_test = test[features]   
    # Compute probability of observation being in the origin.
    testing_probs[origin] = models[origin].predict_proba(X_test)[:,1]

predicted_origins = testing_probs.idxmax(axis=1)   #select the model with the highest probability as being true and set as result for orgin of car
print(predicted_origins)
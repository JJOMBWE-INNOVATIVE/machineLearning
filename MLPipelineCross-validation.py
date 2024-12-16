# [21:05, 12/8/2024] Jjombwe Nathan: # STEP BY STEP FOR MODELING
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, make_scorer

# Read the data
home_file_path = "C:/Users/isma/Desktop/ml/melb_data.csv"
home_data = pd.read_csv(home_file_path)

# Separate target from predictors
y = home_data.Price
# X = home_data.drop(['Price'], axis=1)
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = home_data[cols_to_use]

# WE USE PIPELINE
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer

# my_pipeline = Pipeline(steps=[('preprocessor', 
#      SimpleImputer()),
#     ('model',
#     RandomForestRegressor
#     (n_estimators=50,
#     random_state=0))
#     ])

# CROSS-VALIDATION
# from sklearn.model_selection import cross_val_score

# # Multiply by -1 since sklearn calculates *negative* MAE
# scores = -1 * cross_val_score(my_pipeline, X, y,
#       cv=5,
#     scoring='neg_mean_absolute_error')

# # print("MAE scores:\n", scores)
# print("Average MAE score (across experiments):")
# print(scores.mean())


# HERE WE WANT TO FIND OUR ESTIMATORS OUR SELVES

def get_score(n_estimators):
    # Create the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(
            n_estimators=n_estimators, random_state=0))
    ])
    
    # Perform cross-validation (scoring based on negative MAE)
    scores = -1 * cross_val_score(pipeline, X, y, 
     cv=3, 
      scoring='neg_mean_absolute_error')
    # Return the average MAE
    return scores.mean()

# Define the range of n_estimators values
n_estimators_values = [50, 100, 150, 200, 250, 300, 350, 400]

# Initialize an empty dictionary to store the results
results = {}

# Loop through each value of n_estimators and calculate the average MAE
for i in n_estimators_values:
    results[i] = get_score(i)

# PLOTING THE ESTIMATORS
# Plot the results
# plt.plot(list(results.keys()), list(results.values()))
# plt.xlabel("n_estimators")
# plt.ylabel("Mean Absolute Error (MAE)")
# plt.title("Performance of Random Forest with Varying n_estimators")
# plt.show()

# Find the n_estimators with the lowest MAE
best_n_estimators = min(results, key=results.get)
print(f"The best value for n_estimators is: {best_n_estimators}")

# Update the final pipeline with the best n_estimators
final_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', RandomForestRegressor(
        n_estimators=best_n_estimators, random_state=0))
])

# Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

# Fit the final model and evaluate on the validation data
final_pipeline.fit(X_train, y_train)
preds = final_pipeline.predict(X_valid)
final_mae = mean_absolute_error(y_valid, preds)

print(f"The MAE of the final model on the validation set is: {final_mae}")










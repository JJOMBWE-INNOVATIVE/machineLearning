
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Read the data
home_file_path = "C:/Users/isma/Desktop/ml/AER_credit_card_data.csv"
card_data = pd.read_csv(home_file_path,true_values = ['yes'], false_values = ['no'])

# Separate target from predictors
y = card_data.card

# Select predictors
X = card_data.drop(['card'], axis=1)

print("Number of rows in the dataset:", X.shape[0])
# print(X.head())

# Since there is no preprocessing, we don't need a pipeline (used anyway as best practice!)
my_pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))
cv_scores = cross_val_score(my_pipeline, X, y, 
    cv=5,
    scoring='accuracy')

print("Cross-validation accuracy: %f" % cv_scores.mean())

expenditures_cardholders = X.expenditure[y]
expenditures_noncardholders = X.expenditure[~y]

print('Fraction of those who did not receive a card and had no expenditures: %.2f' \
      %((expenditures_noncardholders == 0).mean()))
print('Fraction of those who received a card and had no expenditures: %.2f' \
      %(( expenditures_cardholders == 0).mean()))


# DEALING WITH TARGET LEAKAGES
# Drop leaky predictors from dataset
potential_leaks = ['expenditure', 'share', 'active', 'majorcards']
X2 = X.drop(potential_leaks, axis=1)

# Evaluate the model with leaky predictors removed
cv_scores = cross_val_score(my_pipeline, X2, y, 
  cv=5,
  scoring='accuracy')

print("Cross-val accuracy: %f" % cv_scores.mean())
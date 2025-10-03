from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import pandas as pd

df = pd.read_csv("/mnt/datalake/Eta/CollegePlacement.csv")
df['Placement'] = pd.factorize(df['Placement'])[0]
df['Internship_Experience'] = pd.factorize(df['Internship_Experience'])[0]


X = df.drop(["Placement", "College_ID"], axis = 1)

Y = df["Placement"]


logreg = LogisticRegression(random_state=16,  max_iter=1000)

model = logreg.fit(X, Y)
coef_df = {col : coef for col, coef in zip(list(X.columns), model.coef_[0])}
# print(coef_df)
# print(X.columns)
print(f"coefficent: {coef_df}\nintercept: {model.intercept_}")

# Save the model
import joblib
joblib.dump(model, '/mnt/datalake/Eta/AF_college_placement_model.pkl')
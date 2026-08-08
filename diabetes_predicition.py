import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay, ConfusionMatrixDisplay, precision_score, f1_score, recall_score, recall_score, accuracy_score




np.random.seed(42)
df = pd.read_csv(r"D:\python\tut\Scikit\Of_1_Proyecto\Diabetes.csv")

col_nan= [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

df[col_nan] = df[col_nan].replace(0, np.nan)
df.loc[df["SkinThickness"]>60,"SkinThickness"] = np.nan
df.loc[df["BloodPressure"]<30,"BloodPressure"] = np.nan

X = df.drop(labels="Outcome", axis=1)
y = df.Outcome.to_numpy()

columnas_con_nan = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in columnas_con_nan:
    if col in df.columns:
        df[col] = df.groupby('Outcome')[col].transform(lambda x: x.fillna(x.median()))

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2
)

models = {"KNN": KNeighborsClassifier(),
          "Logistic Regression": LogisticRegression(max_iter=400),
          "Random Forest": RandomForestClassifier(),
          "Linear SVC": LinearSVC()}

knn = KNeighborsClassifier(
    n_neighbors=6,
    weights='uniform',
    metric="manhattan"
)

knn.fit(X_train, y_train)

y_preds = knn.predict(X_test)

print(classification_report(y_test, y_preds))



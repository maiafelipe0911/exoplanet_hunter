import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


df = pd.read_csv("cumulative.csv")

y = df['koi_disposition'].apply(lambda x: 0 if x=='FALSE POSITIVE' else 1)


features = [
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
    'koi_period', 'koi_period_err1', 'koi_period_err2',
    'koi_time0bk', 'koi_time0bk_err1', 'koi_time0bk_err2',
    'koi_impact', 'koi_impact_err1', 'koi_impact_err2',
    'koi_duration', 'koi_duration_err1', 'koi_duration_err2',
    'koi_depth', 'koi_depth_err1', 'koi_depth_err2',
    'koi_prad', 'koi_prad_err1', 'koi_prad_err2',
    'koi_teq', 'koi_insol', 'koi_insol_err1', 'koi_insol_err2',
    'koi_model_snr', 
    'koi_steff', 'koi_steff_err1', 'koi_steff_err2',
    'koi_slogg', 'koi_slogg_err1', 'koi_slogg_err2',
    'koi_srad', 'koi_srad_err1', 'koi_srad_err2'
]

X = df[features]

X = X.fillna(X.median())

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state= 1)

model = RandomForestClassifier(random_state=1)
model.fit(X_train, y_train)    
predictions = model.predict(X_val)
print("Model Accuracy:", accuracy_score(y_val, predictions))

importances = model.feature_importances_
feature_names = X.columns

feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\nImportância dos Fatores")
print(feature_importance_df)



cm = confusion_matrix(y_val, predictions)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Isn't a Planet", 'Is a planet'])
disp.plot(cmap='Blues')
plt.title("Where is the model making mistakes?")
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lime.lime_tabular import LimeTabularExplainer

data=pd.read_excel("Lab Session Data.xlsx",sheet_name="marketing_campaign")
data=data.drop(columns=["ID","Dt_Customer"])
data=data.dropna(subset=["Response"])

X=data.drop(columns=["Response"]).copy()
y=data["Response"]

for col in X.select_dtypes(include="object").columns:
    X[col]=LabelEncoder().fit_transform(X[col].fillna("missing").astype(str))

X=X.fillna(X.median())

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

def tune_model(model,params):
    pipe=Pipeline([("scaler",StandardScaler()),("classifier",model)])
    search=RandomizedSearchCV(pipe,params,n_iter=10,cv=4,scoring="accuracy",random_state=42,n_jobs=-1)
    search.fit(X_train,y_train)
    return search

def evaluate_model(model):
    pred=model.predict(X_test)
    return [accuracy_score(y_test,pred),precision_score(y_test,pred,zero_division=0),recall_score(y_test,pred,zero_division=0),f1_score(y_test,pred,zero_division=0)]

print("A2\n")

p=tune_model(Perceptron(random_state=42),{
    "classifier__penalty":[None,"l1","l2","elasticnet"],
    "classifier__alpha":[0.0001,0.001,0.01,0.1],
    "classifier__max_iter":[500,1000,1500],
    "classifier__eta0":[0.001,0.01,0.1,1.0]
})

print("Perceptron Results")
print("Best Parameters")
print(p.best_params_)
print("Best CV Accuracy")
print(p.best_score_)
print("Test Accuracy")
print(accuracy_score(y_test,p.best_estimator_.predict(X_test)))

m=tune_model(MLPClassifier(max_iter=500,random_state=42),{
    "classifier__hidden_layer_sizes":[(50,),(100,),(50,50),(100,50)],
    "classifier__activation":["relu","tanh"],
    "classifier__alpha":[0.0001,0.001,0.01],
    "classifier__solver":["adam","sgd"]
})

print("\nMLP Results")
print("Best Parameters")
print(m.best_params_)
print("Best CV Accuracy")
print(m.best_score_)
print("Test Accuracy")
print(accuracy_score(y_test,m.best_estimator_.predict(X_test)))
print("\n")


print("A3\n")

models={
    "Perceptron":Perceptron(random_state=42),
    "MLP":MLPClassifier(max_iter=500,random_state=42),
    "SVM":SVC(),
    "Decision Tree":DecisionTreeClassifier(random_state=42),
    "Random Forest":RandomForestClassifier(n_estimators=100,random_state=42),
    "CatBoost":CatBoostClassifier(verbose=0,random_state=42),
    "AdaBoost":AdaBoostClassifier(n_estimators=50,random_state=42),
    "XGBoost":XGBClassifier(eval_metric="logloss",random_state=42),
    "Naive Bayes":GaussianNB()
}

results=[]

for name,model in models.items():
    pipe=Pipeline([("scaler",StandardScaler()),("classifier",model)])
    pipe.fit(X_train,y_train)
    metrics=evaluate_model(pipe)
    results.append([name]+metrics)

results_df=pd.DataFrame(results,columns=["Model","Accuracy","Precision","Recall","F1 Score"])
print(results_df.round(4))
results_df.to_csv("A3_Classifier_Results.csv",index=False)
print("Results saved to A3_Classifier_Results.csv")
print("\n")


print("A4\n")

model=Pipeline([
    ("scaler",StandardScaler()),
    ("classifier",RandomForestClassifier(n_estimators=100,random_state=42))
])

model.fit(X_train,y_train)
pred=model.predict(X_test)

print("Pipeline Accuracy")
print(accuracy_score(y_test,pred))
print("\nClassification Report")
print(classification_report(y_test,pred,zero_division=0))
print("\n")


print("A5\n")

explainer=LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=["No Response","Response"],
    mode="classification",
    random_state=42
)

exp=explainer.explain_instance(
    X_test.iloc[0].values,
    model.predict_proba,
    num_features=10
)

print("Actual Response")
print(y_test.iloc[0])
print("Predicted Response")
print(model.predict(X_test.iloc[[0]])[0])
print("LIME Explanation")
print(exp.as_list())

exp.save_to_file("A5_LIME_Explanation.html")
print("Explanation saved to A5_LIME_Explanation.html")

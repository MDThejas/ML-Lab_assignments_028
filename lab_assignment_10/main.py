
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data=pd.read_excel("Lab Session Data.xlsx",sheet_name="marketing_campaign")
data=data.drop(columns=["ID","Dt_Customer"])
data=data.dropna(subset=["Response"])

X=data.drop(columns=["Response"]).copy()
y=data["Response"]

for col in X.select_dtypes(include="object").columns:
    X[col]=LabelEncoder().fit_transform(X[col].fillna("missing").astype(str))

X=X.fillna(X.median())


print("A2\n")

corr_matrix=X.corr()

print("Feature Correlation Matrix")
print(corr_matrix.round(2))

plt.figure(figsize=(14,10))
sns.heatmap(corr_matrix,annot=True,cmap="coolwarm",fmt=".2f",linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("A2_correlation_heatmap.png",dpi=300)
plt.show()
print("\n")


print("A3\n")

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

pca=PCA()
pca.fit(X_train_scaled)

cum_var=np.cumsum(pca.explained_variance_ratio_)
n_components=np.argmax(cum_var>=0.99)+1

pca=PCA(n_components=n_components)
X_train_pca=pca.fit_transform(X_train_scaled)
X_test_pca=pca.transform(X_test_scaled)

print("Original Number of Features:",X_train.shape[1])
print("Number of PCA Components:",pca.n_components_)
print("Explained Variance Retained:",round(sum(pca.explained_variance_ratio_)*100,2),"%")

models={
    "Decision Tree":DecisionTreeClassifier(random_state=42),
    "Random Forest":RandomForestClassifier(n_estimators=100,random_state=42),
    "SVM":SVC()
}

original_results={}
pca_results={}

for name,model in models.items():
    model.fit(X_train_scaled,y_train)
    pred=model.predict(X_test_scaled)
    original_results[name]=accuracy_score(y_test,pred)*100

for name,model in models.items():
    model.fit(X_train_pca,y_train)
    pred=model.predict(X_test_pca)
    pca_results[name]=accuracy_score(y_test,pred)*100

results=pd.DataFrame({
    "Model":list(models.keys()),
    "Original Features Accuracy (%)":[original_results[name] for name in models],
    "PCA Accuracy (%)":[pca_results[name] for name in models]
})

print("\nClassification Performance Comparison")
print(results.round(2).to_string(index=False))
results.to_csv("A3_PCA_Results.csv",index=False)


plt.figure(figsize=(8,5))
plt.plot(range(1,len(cum_var)+1),cum_var*100,marker="o")
plt.axhline(y=99,linestyle="--",label="99% variance")
plt.axvline(x=n_components,linestyle="--",label="Selected Components")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.title("PCA - Cumulative Explained Variance")
plt.legend()
plt.tight_layout()
plt.savefig("A3_pca_variance.png",dpi=300)
plt.show()


results_plot=results.set_index("Model")
results_plot.plot(kind="bar",figsize=(9,5))
plt.ylabel("Accuracy (%)")
plt.title("Original Features vs PCA")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("A3_accuracy_comparison.png",dpi=300)
plt.show()

import pandas as pd
import math 
import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import GridSearchCV


data=pd.read_excel("Lab Session Data.xlsx",sheet_name="marketing_campaign")
data=data.fillna(data.mean(numeric_only=True))

def binning(col,b):
    min_value=col.min()
    max_value=col.max()
    width=(max_value-min_value)/b
    binned_data=[]
    for value in col:
        bin_no=int((value-min_value)/width)
        if bin_no==b:
            bin_no=bin_no-1
        binned_data.append(bin_no)
    return binned_data

def cal_entropy(bin_col):
    total=len(bin_col)
    counts={}
    for value in bin_col:
        if value not in counts:
            counts[value]=0
        counts[value]+=1
    entropy=0
    for count in counts.values():
        p=count/total
        entropy=entropy-(p*math.log2(p))
    return entropy

print("A1\n")
test_col=data["Response"]
bin_column=binning(test_col,4)
ent_value=cal_entropy(bin_column)
print("entropy of response column is",ent_value)
print("\n")

def cal_gini(col):
    total=len(col)
    counts={}
    for value in col:
        if value not in counts:
            counts[value]=0
        counts[value]+=1
    gini = 1
    for count in counts.values():
        p=count/total
        gini=gini-(p*p)
    return gini

print("A2\n")
gini_value = cal_gini(data["Response"])
print("Gini Index for response col is",gini_value)
print("\n")

def information_gain(feature,target):
    total=len(feature)
    parent_entropy=cal_entropy(target)
    unique=[]
    for value in feature:
        if value not in unique:
            unique.append(value)
    weighted_entropy=0
    for value in unique:
        subset=[]
        for i in range(total):
            if feature[i]==value:
                subset.append(target[i])
        entropy=cal_entropy(subset)
        weight=len(subset)/total
        weighted_entropy=weighted_entropy+(weight*entropy)
    gain=parent_entropy-weighted_entropy
    return gain

print("A3\n")

target=list(data["Response"])
features=["Income","Recency","MntWines","MntFruits","NumWebPurchases"]
max_gain=-1
root_node=""
for feature in features:
    feature_data=binning(data[feature],4)
    gain=information_gain(feature_data,target)
    print(feature,gain)
    if gain>max_gain:
        max_gain=gain
        root_node=feature
print("\n")
print("Root Node =",root_node)
print("\n")
print("Information Gain =",max_gain)
print("\n")

def binning_with_type(col,b=4,bin_type="equal_width"):
    if bin_type=="equal_width":
        min_value=col.min()
        max_value=col.max()
        width=(max_value-min_value)/b
        binned=[]
        for value in col:
            bin_no=int((value-min_value)/width)
            if bin_no==b:
                bin_no=b-1
            binned.append(bin_no)
        return binned

    elif bin_type=="frequency":
        sorted_col=sorted(col)
        size=len(col)//b
        binned=[]
        for value in col:
            pos=sorted_col.index(value)
            bin_no=pos//size
            if bin_no==b:
                bin_no=b-1
            binned.append(bin_no)
        return binned

print("A4\n")
width_bin=binning_with_type(data["Income"],4,"equal_width")
freq_bin=binning_with_type(data["Income"],4,"frequency")
print("Equal Width Binning")
print(width_bin)
print("\n")
print("Equal Frequency Binning")
print(freq_bin)
print("\n")


def build_tree(data,features,target_col):
    target=list(data[target_col])
    max_gain=-1
    root=""
    for feature in features:
        if feature!=target_col:
            feature_data=binning(data[feature],4)
            gain=information_gain(feature_data,target)
            if gain>max_gain:
                max_gain=gain
                root=feature
    root_data=binning(data[root],4)
    tree={}
    tree[root]={}
    unique=[]
    for value in root_data:
        if value not in unique:
            unique.append(value)
    for value in unique:
        class_count={}
        for i in range(len(root_data)):
            if root_data[i]==value:
                response=target[i]
                if response not in class_count:
                    class_count[response]=0
                class_count[response]+=1
        majority=max(class_count,key=class_count.get)
        tree[root][value]=majority
    return tree

print("A5\n")
features=["Income","Recency","MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds","NumDealsPurchases","NumWebPurchases"]
tree=build_tree(data,features,"Response")
print("Decision Tree")
print(tree)
print("\n")


print("A6\n")
X=data[features]
y=data["Response"]
model=DecisionTreeClassifier(criterion="entropy",random_state=42)
model.fit(X,y)
plt.figure(figsize=(15,8))
plot_tree(model,feature_names=features,class_names=["0","1"],filled=True)
plt.title("Decision Tree")
plt.show()
print("\n")


print("A7\n")
X=data[["Income","Recency"]]
y=data["Response"]
clf=DecisionTreeClassifier(criterion="entropy",random_state=42)
clf.fit(X,y)

DecisionBoundaryDisplay.from_estimator(clf,X,response_method="predict",xlabel="Income",ylabel="Recency",alpha=0.5)
plt.scatter(X["Income"],X["Recency"],c=y,edgecolor="black",s=20)
plt.title("Decision Boundary of Decision Tree")
plt.show()
print("\n")

print("A8\n")
X=data[features]
y=data["Response"]
model=DecisionTreeClassifier()
param_grid={"criterion":["entropy","gini"],"max_depth":[3,5,7,10],"min_samples_split":[2,5,10]}
grid_search=GridSearchCV(estimator=model,param_grid=param_grid,cv=5,n_jobs=-1)
grid_search.fit(X,y)

print("Best Parameters\n")
print(grid_search.best_params_)
print("Best Score\n")
print(grid_search.best_score_)
print("\n")
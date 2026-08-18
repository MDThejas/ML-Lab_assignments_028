import pandas as pd
import numpy as np 
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  
import matplotlib.pyplot as plt

print("A1\n")
data=pd.read_excel("Lab Session Data.xlsx",sheet_name="marketing_campaign")

# distance function 
def eu_distance(a,b):
    total=0
    for i in range(len(a)):
        total+=(a[i]-b[i])**2
    return total ** 0.5

#missing values
def fill_missing(data):
    numeric_cols=data.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        data[col]=data[col].fillna(data[col].mean())
    return data

# encoding data
def label_encoding(column):
    unique_vals=column.unique()
    encoding={}
    for i,j in enumerate(unique_vals):
        encoding[j]=i
    lst=[]
    for i in column:
        lst.append(encoding[i])
    return lst

def one_hot_encoding(column):
    unique_vals=column.unique()
    lst=[]
    for i in column:
        row=[]
        for j in unique_vals:
            if i==j:
                row.append(1)
            else:
                row.append(0)
        lst.append(row)
    return pd.DataFrame(lst,columns=unique_vals)

# sorting function 
def bubble_sort(arr):
    n=len(arr)
    for i in range(n):
        for j in range(0,n-i-1):
            if arr[j][0]>arr[j+1][0]:
                arr[j],arr[j+1]=arr[j+1],arr[j]
    return arr

def selection_sort(arr):
    n=len(arr)
    for i in range(n):
        min_idx=i
        for j in range(i+1,n):
            if arr[j][0]<arr[min_idx][0]:
                min_idx=j
        arr[i],arr[min_idx]=arr[min_idx],arr[i]
    return arr


def insertion_sort(arr):
    for i in range(1,len(arr)):
        key=arr[i]
        j=i-1
        while j>=0 and arr[j][0]>key[0]:
            arr[j+1]=arr[j]
            j-=1
        arr[j+1] =key
    return arr

#To find distance btw test point to all others points and sort them  
def get_neighbours(X_train,y_train,test_row,k,sort_type):
    dist=[]
    for i in range(len(X_train)):
        d=eu_distance(test_row,X_train[i])
        dist.append([d,y_train[i]])
    if sort_type=="bubble":
        dist=bubble_sort(dist)
    elif sort_type=="selection":
        dist=selection_sort(dist)
    else:
        dist=insertion_sort(dist)
    first_k_values=[]
    for i in range(k):
        first_k_values.append(dist[i])
    return first_k_values

#find the class with most number off repetation 
def most_number_class(arr):
    labels=[]
    for i in arr:
        labels.append(i[1])   
    count=Counter(labels)
    max_count=0
    max_class=-1
    for key in count:
        if count[key]>max_count:
            max_count=count[key]
            max_class=key
    return max_class

def weighted_class(arr):
    weights={}
    for i in arr:
        dist=i[0]
        label=i[1]
        weight=1/(dist)
        if label not in weights:
            weights[label]=0
        weights[label]+=weight
    max_weight=0
    w_class=-1
    for i in weights:
        if weights[i]>max_weight:
            max_weight=weights[i]
            w_class=i
    return w_class

marital_data=one_hot_encoding(data["Marital_Status"])
education_data=one_hot_encoding(data["Education"])
data=pd.concat([data,marital_data,education_data],axis=1)
data=data.drop(["Marital_Status","Education"],axis=1)
data=fill_missing(data)
X=data[["Income","Recency","MntWines","MntFruits","MntMeatProducts","NumWebPurchases","NumStorePurchases"]].values
y=data["Response"].values
test_row=X[0]
neighbours=get_neighbours(X,y,test_row,3,"bubble")
print("k nearest neighbhours are\n",neighbours)
result=most_number_class(neighbours)
print("Max_Class =",result)
print("\n")

print("A2\n")
neighbours=get_neighbours(X,y,test_row,3,"bubble")
print("k nearest neighbhours are\n",neighbours)
w_result=weighted_class(neighbours)
print("max_weighted_class is =",w_result)
print("\n")


print("A3\n")
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
print("X_train",X_train)
print("X_test",X_test)
print("y_train",y_train)
print("y_test",y_test)
print("\n")

print("A4\n")
print("Training knn classifier")
neigh=KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,y_train)
print("\n")

print("A5\n")
print("Test the accuracy of the kNN using the test set obtained \n")
accuracy=neigh.score(X_test, y_test) 
print(accuracy,"\n")

print("A6\n")
print("Use the predict() function to study the prediction behavior of the classifier for test vectors\n")
label=neigh.predict(X_test) 
print(label,"\n")

print("A7\n")
def fit(x_train,y_train):
    value={}
    value["x_train"]=x_train
    value["y_train"]=y_train
    return value
value=fit(X_train,y_train)

def predict(value,test_row,k):
    x_train=value["x_train"]
    y_train=value["y_train"]
    neighbours=get_neighbours(x_train,y_train,test_row,k,"bubble")
    label=most_number_class(neighbours)
    return label

def w_predict(value,test_row,k):
    x_train=value["x_train"]
    y_train=value["y_train"]
    neighbours=get_neighbours(x_train,y_train,test_row,k,"bubble")
    label=weighted_class(neighbours)
    return label

k=3
test_row=X_test[0]
prediction=predict(value,test_row,k)
print("output off given sample data is ",prediction)

def score(value,X_test,y_test,k):
    correct=0
    for i in range(len(X_test)):
        pred=predict(value,X_test[i],k)
        if pred==y_test[i]:
            correct+=1
    ac=correct/len(X_test)
    return ac

def W_score(value,X_test,y_test,k):
    correct=0
    for i in range(len(X_test)):
        pred=w_predict(value,X_test[i],k)
        if pred==y_test[i]:
            correct+=1
    ac=correct/len(X_test)
    return ac

score_value=score(value,X_test,y_test,k)
print("accuracy off model is",score_value)
print("\n")

print("A8\n")
k_values=[]
my_accu_function=[]
sk_accu_built_in=[]

for k in range(1,10):
    neigh=KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train,y_train)
    acc1=neigh.score(X_test,y_test)
    acc2=score(value,X_test,y_test,k)
    k_values.append(k)
    sk_accu_built_in.append(acc1)
    my_accu_function.append(acc2)
print("k values are:",k_values)
print("my accuracy are",my_accu_function)
print("sklearn accuracy are",sk_accu_built_in)

plt.plot(k_values,my_accu_function,marker="-")
plt.plot(k_values,sk_accu_built_in,marker="*")
plt.xlabel("k value")
plt.ylabel("accuracy")
plt.title("my kNN vs built_in kNN")
plt.show()
print("\n")

print("A9\n")
W_k_values=[]
my_W_accu_function=[]
sk_W_accu_built_in=[]

for k in range(1,10):
    neigh=KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train,y_train)
    acc1=neigh.score(X_test,y_test)
    acc2=W_score(value,X_test,y_test,k)
    W_k_values.append(k)
    sk_W_accu_built_in.append(acc1)
    my_W_accu_function.append(acc2)
print("k values are:",W_k_values)
print("my accuracy are",my_W_accu_function)
print("sklearn accuracy are",sk_W_accu_built_in)

plt.plot(W_k_values,my_W_accu_function,marker="-")
plt.plot(W_k_values,sk_W_accu_built_in,marker="*")
plt.xlabel("k value")
plt.ylabel("accuracy")
plt.title("my W_kNN vs built_in kNN")
plt.show()
print("\n")
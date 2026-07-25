import pandas as pd
import numpy as np 
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski

# A1
# nominal : data with are just like names or category but no specfic order
# ordinal : data with some spefic order
# interval : data with order + equal diff
# ratio : data with order + equal diff + here zero has its own value | true value
# nominal: ID,martial_status,"all 0\1 or true or flase"
# ordinal: education
# interval: year,Dt_Customer
# ratio: income,Kidhome,Teenhome

data=pd.read_excel("Lab Session Data.xlsx",sheet_name="marketing_campaign",usecols="A:AC")
a=[1,2,4]
b=[4,5,6]
p=3

print("A2\n")

def label_encoding(column):
    unique_vals=column.unique()
    encoding={}
    for i,j in enumerate(unique_vals):
        encoding[j]=i
    lst=[]
    for i in column:
        lst.append(encoding[i])
    return lst

def one_hot_encoding(col):
    col=data["Marital_Status"]
    uni_value=col.unique()
    lst=[]
    for i in col:
        row=[]
        for j in uni_value:
            if i==j:
                row.append(1)
            else:
                row.append(0)
        lst.append(row)
    return pd.DataFrame(lst, columns=uni_value)

encoded=label_encoding(data["Education"])
print("Label encoding for ordinal data type:Education\n",encoded)
print("\n")
encoded_one=one_hot_encoding(data["Marital_Status"])
print("one hot encoding for nominal data type:Marital_status\n",encoded_one)
print("\n")

print("A3\n")
def A3(data):
    orignal_size=data.shape
    data["Education"]=label_encoding(data["Education"])
    after_L_encode=data.shape
    martial=one_hot_encoding(data["Marital_Status"])
    data=data.drop("Marital_Status",axis=1)
    data=pd.concat([data,martial],axis=1)
    after_one_h=data.shape
    return orignal_size,after_L_encode,after_one_h

A,B,C=A3(data)
print("dimension before any operation",A)
print("dimension after label encoding is",B)
print("dimension after one hot encoding is",C)


print("A4\n")

def A4(a,b,p):
    total=0
    for i,j in zip(a,b):
        if i>j:
            total+=(i-j)**p
        else:
            total+=(j-i)**p
    total_dist=total**(1/p)
    return total_dist

min=A4(a,b,p)
print("Minkowski Distance at p=3",min)
man=A4(a,b,1)
print("Manhattan Distance is given",man)
eud=A4(a,b,2)
print("Euclidean Distance is given",eud)
print("\n")

print("A5\n")
def A5(v1,v2):
    p_value=[]
    distance_p=[]
    for i in range(1,11):
        dist=A4(v1,v2,i)
        p_value.append(i)
        distance_p.append(dist)
    plt.plot(p_value,distance_p)
    plt.title("graph btw p_value and distance")
    plt.grid(True)
    plt.xlabel("p_value")
    plt.ylabel("distance_p")
    plt.show()
    return p_value,distance_p

data_frame=data.select_dtypes(include=["int64","float64"])
v1=data_frame.iloc[0].to_numpy()
v2=data_frame.iloc[1].to_numpy()
p_value,dist_p=A5(v1,v2)
print("p_values are",p_value)
print("distance corresponding to P_values are",dist_p)
print("\n")

print("A6\n")
def A6(vec1,vec2,p_v):
    own_method=A4(vec1,vec2,p_v)
    built_in_method=minkowski(vec1,vec2,p_v)
    return own_method,built_in_method
d1,d2=A6(v1,v2,3)
diff=d1-d2

if diff==0:
    print("they are equal",diff)
else:
    print("they are diff",diff)
print("\n")

print("A7\n")

def dot_product(vec1,vec2):
    total=0
    for i,j in zip(vec1,vec2):
        total+=i*j
    return total
def euc_Norm(vec):
    total=0
    for i in vec:
        total+=i**2
    length=total**2
    return length

def A7(vec1,vec2):
    own_meth_dot_product=dot_product(vec1,vec2)
    built_in_meth_prod=np.dot(vec1,vec2)
    own_length_v1=euc_Norm(vec1)
    built_in_len_v1=np.linalg.norm(vec1)
    own_length_v2=euc_Norm(vec2)
    built_in_len_v2=np.linalg.norm(vec2)
    return own_meth_dot_product,built_in_meth_prod,own_length_v1,built_in_len_v1,own_length_v2,built_in_len_v2
vec1=[2,3,4,5]
vec2=[5,6,7,8]
own_dot,b_dot,own_v1,b_v1,own_v2,b_v2=A7(vec1,vec2)
print("dot product off vectors v1 and v2 by own method is",own_dot)
print("dot product off vectors v1 and v2 by built in method is",b_dot)
if own_dot==b_dot:
    print("they are equal")
else:
    print("not_equal")
print("length of v1 by own method",own_v1)
print("length of v1 by built in method is",b_v1)
print("length of v2 by built in method is",b_v2)
print("length of v2 by own method",own_v2)
print("\n")

print("A8\n")
def mean_n(v):
    m=sum(v)/len(v)
    return m
def variance_std_d(v):
    total=0
    avg=mean_n(v)
    for i in v:
        total+=(i-avg)**2
    var=total/len(v)
    std=var**(1/2)
    return var,std

def A8(data):
    num_data=data.select_dtypes(include=["int64","float64"])
    columns=[]
    means=[]
    vars=[]
    stds=[]
    for col in num_data.columns:
        value=num_data[col]
        mean=mean_n(value)
        var,std=variance_std_d(value)
        columns.append(col)
        means.append(mean)
        vars.append(var)
        stds.append(std)
    return columns,means,vars,stds

column,m,v,s=A8(data)
print("mean_var_std of all the numeric data columns are:\n")
print(column,"\n")
print(m,"\n")
print(v,"\n")
print(s,"\n")
print("\n")

print("A9\n")
def A9(data):
    num_data=data.select_dtypes(include=["int64","float64"])
    mean=np.mean(num_data,axis=0)
    std=np.std(num_data,axis=0)
    return mean,std
mean_built_in,std_built_in=A9(data)
print("mean of all colums with numpy module is \n",mean_built_in)
print("std of all columns with numpy module is \n",std_built_in)

for i in range(len(column)):
    print(column[i])
    print("own_mean",m[i])
    print("numpy_mean",mean_built_in.iloc[i])
    print("own_std",s[i])
    print("numpy_std",std_built_in.iloc[i])
print("\n")

print("A10\n")
def A10(data,column):
    value=data[column].dropna()
    bin=10
    hist,bin=np.histogram(value,bin)
    mean=np.mean(value)
    var=np.var(value)
    plt.hist(value,bin)
    plt.title("frequency vs ranges")
    plt.xlabel("range")
    plt.ylabel("frequency")
    plt.show()
    return hist,bin,mean,var
hist,bin,np_mean,np_var=A10(data,"Income")
print("ranges are given by",bin)
print("histogram is given",hist)
print("mean of column is",np_mean)
print("var of columb is",np_var)
print("\n")

print("A11\n")


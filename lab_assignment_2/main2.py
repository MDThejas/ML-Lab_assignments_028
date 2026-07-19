import pandas as pd
import numpy as np
import seaborn as sns
import time 
import matplotlib.pyplot as plt
print("5th question")
def A5():
    data = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI",usecols="A:AE")
    data=data.replace({"t":1,"f":0})
    binary=data.loc[:, data.isin([0,1]).all()]
    row1=binary.iloc[0]
    row2=binary.iloc[1]
    f11=f01=f10=f00=0
    for a,b in zip(row1,row2):
        if a==1 and b==1:
            f11+=1
        elif a==0 and b==1:
            f01+=1
        elif a==1 and b==0:
            f10+=1
        else:
            f00+=1
    jc= f11/(f11+f10+f01)
    smc= (f11+f00)/(f11+f01+f10+f00)
    return jc ,smc

jc_value,smc_value=A5()
print("value of Jc is",jc_value)
print("value of smc is",smc_value)

print("\n")
print("6th question")

def A6():
    data=pd.read_excel("Lab Session Data.xlsx",sheet_name="thyroid0387_UCI",usecols="A:AE")
    data=data.replace({"t":1,"f":0})
    data=data.replace("?",np.nan)
    for col in data.columns:
        data[col]=pd.factorize(data[col])[0]
    data=data.fillna(0)
    row1=data.iloc[0].to_numpy(dtype=float)
    row2=data.iloc[1].to_numpy(dtype=float)
    dot_product=np.dot(row1,row2)
    mag1=np.linalg.norm(row1)
    mag2=np.linalg.norm(row2)
    cosine_s=dot_product/(mag1*mag2)
    return cosine_s

similarity=A6()
print("Cosine Similarity is =",similarity)


def A7():
    data=pd.read_excel("Lab Session Data.xlsx",sheet_name="thyroid0387_UCI",usecols="A:AE")
    data=data.replace({"t":1,"f":0})
    data=data.replace("?",np.nan)

    for col in data.columns:
        data[col]=pd.factorize(data[col])[0]
    data=data.fillna(0)
    data=data.iloc[:20]

    jc=np.zeros((20,20))
    smc=np.zeros((20,20))
    cos=np.zeros((20,20))

    for i in range(20):
        for j in range(20):
            row1=data.iloc[i].to_numpy(dtype=float)
            row2=data.iloc[j].to_numpy(dtype=float)
            f11=f10=f01=f00=0

            for a,b in zip(row1,row2):
                if a==1 and b==1:
                    f11+=1
                elif a==1 and b==0:
                    f10+=1
                elif a==0 and b==1:
                    f01+=1
                elif a==0 and b==0:
                    f00+=1
            if(f11+f10+f01)!=0:
                jc[i][j]=f11/(f11+f10+f01)
            smc[i][j]=(f11+f00)/(f11+f10+f01+f00)
            cos[i][j]=np.dot(row1,row2)/(np.linalg.norm(row1)*np.linalg.norm(row2))

    plt.figure(figsize=(8,6))
    sns.heatmap(jc)
    plt.title("Jaccard Coefficient")
    plt.show()

    plt.figure(figsize=(8,6))
    sns.heatmap(smc)
    plt.title("Simple Matching Coefficient")
    plt.show()

    plt.figure(figsize=(8,6))
    sns.heatmap(cos)
    plt.title("Cosine Similarity")
    plt.show()

print("\n")
print("7th question")
A7()

print("\n")
print("8th question")

def A8():
    data=pd.read_excel("Lab Session Data.xlsx",sheet_name="thyroid0387_UCI",usecols="A:AE")
    data=data.replace("?",np.nan)
    mean_cols=["T3","T4U"]
    median_cols=["age","TSH","TT4","FTI","TBG"]
    for col in mean_cols:
        data[col]=pd.to_numeric(data[col]) # doesnt contain outliers 
        data[col]=data[col].fillna(data[col].mean())
    for col in median_cols: # contains outliers so median wont get affected
        data[col]=pd.to_numeric(data[col])
        data[col]=data[col].fillna(data[col].median())
    for col in data.columns: # categorical attributes we can use mode 
        if col not in mean_cols and col not in median_cols:
            data[col]=data[col].fillna(data[col].mode()[0])
    return data

value=A8()
print(value.head())

print("\n")
print("9th question")

def A9():
    data=pd.read_excel("Lab Session Data.xlsx",sheet_name="thyroid0387_UCI",usecols="A:AE")
    data=data.replace("?",np.nan)
    numeric_cols=["age","TSH","T3","TT4","T4U","FTI","TBG"]
    for col in numeric_cols:
        data[col]=pd.to_numeric(data[col])
        data[col]=(data[col]-data[col].min())/(data[col].max()-data[col].min())
    return data

normalize_value=A9()
print(normalize_value.head())
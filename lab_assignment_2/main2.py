import pandas as pd
import numpy as np
import time 
import matplotlib.pyplot as plt

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


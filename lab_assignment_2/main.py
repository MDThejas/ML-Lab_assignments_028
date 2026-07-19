import pandas as pd
import numpy as np
import time 
import matplotlib.pyplot as plt


def A1():
    purchase = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data")
    X = purchase[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].to_numpy() #will convert the dataframe into numpy array
    y = purchase["Payment (Rs)"].to_numpy()
    rank = np.linalg.matrix_rank(X)
    cost = np.linalg.pinv(X) @ y
    return rank, cost

rank, cost = A1()

print("Rank:", rank)
print("Candies:", cost[0])
print("Mangoes:", cost[1])
print("Milk Packets:", cost[2])
print("\n")

print("2nd question")
def A2():
    purchase = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data",usecols="A:E")

    status = []

    for payment in purchase["Payment (Rs)"]:
        if payment > 200:
            status.append("RICH")
        else:
            status.append("POOR")

    purchase["Customer Status"] = status

    return purchase

purchase = A2()
print(purchase)
print("\n")
print("3rd question")
def A3():
    stock=pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price",usecols="A:I")
    inp=stock["Price"].to_numpy()
    mean=np.mean(inp)
    var=np.var(inp)
    return mean , var ,inp

def my_mean(arr):
    n=len(arr)
    m=sum(arr)
    mean=m/n
    return mean

def my_var(arr): # sum off all (x-x')^2 divide by n 
    m=my_mean(arr)
    total=0
    for x in arr:
        total+=(x-m)**2
    return total/len(arr)

def compare(arr):
    np_mean=np.mean(arr)
    cal_mean=my_mean(arr)
    np_var=np.var(arr)
    cal_var=my_var(arr)

    np_mean_array=[]
    for i in range(10):
        start=time.perf_counter()
        np.mean(arr)
        end=time.perf_counter()
        np_mean_array.append(end-start)
    
    avg_numpy_mean_time=sum(np_mean_array)/10

    np_var_array=[]
    for i in range(10):
        start=time.perf_counter()
        np.var(arr)
        end=time.perf_counter()
        np_var_array.append(end-start)
    
    avg_numpy_var_time=sum(np_var_array)/10

    cal_mean_array=[]
    for i in range(10):
        start=time.perf_counter()
        my_mean(arr)
        end=time.perf_counter()
        cal_mean_array.append(end-start)
    
    avg_cal_mean_time=sum(cal_mean_array)/10

    cal_var_array=[]
    for i in range(10):
        start=time.perf_counter()
        my_var(arr)
        end=time.perf_counter()
        cal_var_array.append(end-start)
    
    avg_cal_var_time=sum(cal_var_array)/10

    return (np_mean,np_var,cal_mean,cal_var,avg_numpy_mean_time,avg_numpy_var_time,avg_cal_mean_time,avg_cal_var_time)

def wed_data():
    stock=pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price",usecols="A:I")
    population_mean=np.mean(stock["Price"])
    stock_wed=stock[stock["Day"]=="Wed"]
    sample_mean=np.mean(stock_wed["Price"])
    return population_mean,sample_mean

def apr_data():
    stock=pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price",usecols="A:I")
    population_mean=np.mean(stock["Price"])
    stock_wed=stock[stock["Month"]=="Apr"]
    sample_mean=np.mean(stock_wed["Price"])
    return population_mean,sample_mean

def profit_loss():
    stock=pd.read_excel("Lab Session Data.xlsx",sheet_name="IRCTC Stock Price",usecols="A:I")
    total_days=len(stock)
    loss_days=len(stock[stock["Chg%"]<0])
    profit_days=len(stock[stock["Chg%"]>0])
    loss_prob=loss_days/total_days
    profit_prob=profit_days/total_days
    return loss_prob, profit_prob

def profit_on_wed():
    stock=pd.read_excel("Lab Session Data.xlsx",sheet_name="IRCTC Stock Price",usecols="A:I")
    total_days=len(stock)
    wed_profit=len(stock[(stock["Day"]=="Wed") & (stock["Chg%"]>0)])
    prob=wed_profit/total_days
    return prob

def conditional_wed():
    stock=pd.read_excel("Lab Session Data.xlsx",sheet_name="IRCTC Stock Price",usecols="A:I")
    stock_wed=stock[stock["Day"]=="Wed"]
    total_days=len(stock_wed)
    profit_days=len(stock_wed[stock_wed["Chg%"]>0])
    profit_p=profit_days/total_days
    return profit_p

def scatter_plot():
    stock = pd.read_excel("Lab Session Data.xlsx",sheet_name="IRCTC Stock Price",usecols="A:I")
    plt.scatter(stock["Day"],stock["Chg%"])
    plt.title("Scatter plot")
    plt.xlabel("Day")
    plt.ylabel("Chg%")
    plt.show()

def mean_variance():
    thy_data=pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI",usecols="A:AE")
    thy_data=thy_data.replace("?",np.nan)
    numeric_cols=["age","TSH","T3","TT4","T4U","FTI","TBG"]
    thy_data[numeric_cols]=thy_data[numeric_cols].apply(pd.to_numeric)
    mean=thy_data[numeric_cols].mean()
    var=thy_data[numeric_cols].var()
    return mean,var



mean,var,arr=A3()
print("\n")
print("3a")
print("numpy_mean is ",mean)
print("numpy_var is",var)
print("\n")
print("3b")
my_m=my_mean(arr)
print("cal_mean is ",my_m)

my_v=my_var(arr)
print("cal_var is",my_v)

(np_mean,np_var,cal_mean,cal_var,avg_numpy_mean_time,avg_numpy_var_time,avg_cal_mean_time,avg_cal_var_time)=compare(arr)

print("mean difference is",abs(np_mean-cal_mean))
print("var difference is",abs(np_var-cal_var))

print("avg mean time for numpy",avg_numpy_mean_time)
print("avg var time for numpy is ",avg_numpy_var_time)
print("avg mean time for actual is ",avg_cal_mean_time)
print("avg var time for actual is ",avg_cal_var_time)

print("\n")
print("3c")
p_mean,s_mean=wed_data()
print("population mean is based on days",p_mean)
print("sample mean is based on days ",s_mean)

print("\n")
print("3d")
pm_mean,sm_mean=apr_data()
print("population mean is based on month ",pm_mean)
print("sample mean is based on month",sm_mean)

print("\n")
print("3e")
loss_p,profit_p=profit_loss()
print("loss prob is ",loss_p)
print("profit prob is",profit_p)

print("\n")
print("3f")
p=profit_on_wed()
print("Probability of making a profit on Wednesday",p)

print("\n")
print("3g")
profit_p_wed=conditional_wed()
print("profit prob during wednesday is",profit_p_wed)

print("\n")
print("3h")
scatter_plot()

print("\n")
print("4th question")
mean,variance=mean_variance()
print("mean off numeric values are\n",mean)
print("variance off numeric values are\n",variance)
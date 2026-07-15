import pandas as pd
import numpy as np

def A1():
    purchase = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data")
    X = purchase[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].to_numpy()
    y = purchase["Payment (Rs)"].to_numpy()
    rank = np.linalg.matrix_rank(X)
    cost = np.linalg.pinv(X) @ y
    return rank, cost

rank, cost = A1()

print("Rank:", rank)
print("Candies:", cost[0])
print("Mangoes:", cost[1])
print("Milk Packets:", cost[2])

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

def A3():
    stock=pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price",usecols="A:I")
    
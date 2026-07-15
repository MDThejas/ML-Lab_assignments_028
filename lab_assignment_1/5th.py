import random 
def mean_value(number):
    n=len(number)
    m=sum(number)/n
    return m 

def median_value(number):
    n=len(number)
    number.sort()
    if n%2==1:
        median=number[n//2]
    else:
        median=(number[n//2-1] + number[n//2])/2
    return median

def mode_value(number):
    max_count=0
    temp=number[0]
    n=len(number)
    for i in range(n):
        count=0
        for j in range(n):
            if number[j]==number[i]:
                count=count+1
        if count>max_count:
            max_count=count
            temp=number[i]
    return temp

number=[]
for i in range(100):
    temp=random.randint(100,150)
    number.append(temp)
print(number)

mean= mean_value(number)
median=median_value(number)
mode =mode_value(number)

print(mean)
print(median)
print(mode)
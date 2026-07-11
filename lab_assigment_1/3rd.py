def common(list1, list2):
    n1 = len(list1)
    n2 = len(list2)
    total_count = 0

    for i in range(n1):
        temp = list1[i]
        for j in range(n2):
            if list2[j] == temp:
                total_count = total_count + 1
                break
    return total_count


list1 = [1,2,3,4,5,6,9]
list2 = [3,4,7,8,10,9]

num=common(list1, list2)
print(num)
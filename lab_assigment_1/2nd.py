def multi_matrix(mat1, mat2):
    len_row1=len(mat1)
    len_col1=len(mat1[0])
    len_row2=len(mat2)
    len_col2=len(mat2[0])

    if len_col1!=len_row2:
        return "Matrix cant be multiplied"
    result=[]
    for i in range(len_row1):
        row=[]
        for j in range(len_col2):
            total = 0
            for k in range(len_col1):
                total+= mat1[i][k] * mat2[k][j]
            row.append(total)
        result.append(row)
    return result

mat1=[[1,2,3],[4,5,6]]
mat2=[[3,4,5],[5,6,7],[6,7,8]]

result = multi_matrix(mat1, mat2)
print(result)
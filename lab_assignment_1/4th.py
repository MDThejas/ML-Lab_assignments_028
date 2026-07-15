def transpose_mat(mat):
    len_row=len(mat)
    len_col=len(mat[0])
    transpose=[]
    for i in range(len_col):
        row=[]
        for j in range(len_row):
            row.append(mat[j][i])
        transpose.append(row)
    return transpose

mat=[[1,2,3],[4,5,6]]
result=transpose_mat(mat)
print(result)
from IPython.display import clear_output
import numpy as np
import streamlit as st
import pandas as pd


def display_mat(matdis):
    df=pd.DataFrame(matdis)
    st.dataframe(df)


    
def input_ele(Al):
    ele = st.number_input(f"Enter element for {Al}:  ")
    return ele
    
def input_row():
    rows = st.selectbox("select number of rows: ",[1,2,3,4,5,6],1)
    return rows


    
def input_col():
    col = st.selectbox("select number of columns: ",[1,2,3,4,5,6],1)
    return col

        
def fresh_prop():
    # Ask user for number of rows and columns
    row = input_row()
    col = input_col()
    # Initialize empty matrix
    alph='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklwxyz'
    al=0
    matrix = []
    for i in range(int(row)):
        row = []
        for j in range(int(col)):
            row.append(alph[al])
            al+=1
        matrix.append(row)
    #  the resulting matrix
    st.write("Matrix:")
    display_mat(matrix)
    return np.array(matrix)



                                                            #to get info
                                                            #to get order
def order(mat):
    return f'{mat.shape[0]} X {mat.shape[1]}'





                                                            #to get submatrix
def collect_matrix(mat,r,c):
    new_mat=mat
    new_mat=np.delete(new_mat,r,axis=0)
    new_mat=np.delete(new_mat,c,axis=1)
    return new_mat


                                                            #to get det
def det(matrix):                                                                        #determinant
    def collect_matrix(mat,r,c):
        new_mat=mat
        new_mat=np.delete(new_mat,r,axis=0)
        new_mat=np.delete(new_mat,c,axis=1)
        return new_mat

    def modulo1(mat):
        def degen(matgen):
            for m in range(matgen.shape[1]):
                yield ((-1)**m)*matgen[0][m]*modulo2(collect_matrix(matgen,1,m))

        if mat.shape==(2,2):
            det=mat[0][0]*mat[1][1]-mat[0][1]*mat[1][0]
            return det
        else:
            adds=0
            for dmat in degen(mat):
                adds+=dmat

            return adds

    def modulo2(mat):
        def degen(matgen):
            for m in range(matgen.shape[1]):
                yield ((-1)**m)*matgen[0][m]*modulo1(collect_matrix(matgen,1,m))

        if mat.shape==(2,2):
            det=mat[0][0]*mat[1][1]-mat[0][1]*mat[1][0]
            return det
        else:
            adds=0
            for dmat in degen(mat):
                adds+=dmat

            return adds        
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0]==1:
        return f'A matrix of order {matrix.shape} has no determinant'
    else:
        determinant=modulo1(matrix)
        return determinant



                                                                #to get cofactor
def cofactor(matrix):
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0]==1:
        return f'A matrix of order {self.order} has no determinant'
    elif matrix.shape[0] ==matrix.shape[0]==2:
        cof=np.zeros([2,2])
        cof[0][0],cof[1][1],cof[0][1],cof[1][0]=matrix[1][1],matrix[0][0],-matrix[1][0],-matrix[0][1]
        return cof
    for r in range(mat.shape[0]):
        for c in range(mat.shape[0]):
            cof[r][c]=(-1)**(r+c)*mat[r][c]*det(collect_matrix(mat,r,c))     
    return cof
                                                                #to get transpose
def transpose(matrix):                                                                       #transpose
    return np.transpose(matrix)


                                                                #to get adjoint
def adjoint(matrix):                                                                         #adjoint
    return transpose(cofactor(matrix))



                                                                #to get inverse

def inverse(mat):                                                                     #inverse
    if mat.shape[0] != mat.shape[1] or mat.shape[0]==1:
        return f'A matrix of order {order(mat)} has no inverse'
    elif det(mat)==0:
        return f'A matrix of detrminant of zero has infinitive inverse. That is, its elements are infinities.'
    return 
    

                                                                #to get trace

def trace(mat):
    if mat.shape[0] != mat.shape[1] or mat.shape[0]==1:
        return f'A matrix of order {order(mat)} has no trace'
    trace=0
    for i in range(mat.shape[0]):
        trace+=mat[i][i]
    return trace

                                                               #to get rank
def rank(mat):        
    return np.linalg.matrix_rank(mat)


    
                                                                #to get eigenvalues
def eigenvalues(mat):
    if mat.shape[0] != mat.shape[1] or mat.shape[0]==1:
        return f'A matrix of order {order(mat)} has no eigenvalue'
    elif mat.shape[0] > 3:
        return f'A matrix of order {order(mat)} which is greater than 3x3 is not solvable by the code it will be upgraded soon. '
    elif mat.shape[0] == 2:
        try:
            lam1=mat[0][0]+mat[1][1]+((mat[0][0]+mat[1][1])**2 -4*(mat[0][0]*mat[1][1]-mat[0][1]*mat[1][0])    )**0.5        
            lam2=mat[0][0]+mat[1][1]-((mat[0][0]+mat[1][1])**2 -4*(mat[0][0]*mat[1][1]-mat[0][1]*mat[1][0])    )**0.5        
            return lam1/2,lam2/2 
        except:
            return np.linalg.eig(mat)[0]

    elif mat.shape[0] == 3:
        def inspect():
            for i in range(10):
                if a*(i-5)**3+b*(i-5)**2+c*(i-5)+d==0:
                    return i-5
            return "no value"
        def char_eq(pp):
            a=-1
            b=pp[1][1]+pp[2][2]+pp[0][0]
            c=-pp[0][0]*pp[1][1]-pp[0][0]*pp[2][2]+pp[0][1]*pp[1][0]+pp[0][2]*pp[2][0]+pp[2][1]*pp[1][2]-pp[1][1]*pp[2][2]
            d=pp[0][0]*pp[1][1]*pp[2][2]-pp[0][0]*pp[1][2]*pp[2][1]+pp[0][1]*pp[2][0]*pp[1][2]  + pp[0][2]*pp[1][0]*pp[2][1]-pp[0][2]*pp[2][0]*pp[1][1]-pp[0][1]*pp[1][0]*pp[2][2]
            return a,b,c,d
        a,b,c,d=char_eq(mat)
        def quad_sol():
            p=inspect()
            if p=="no value":
                return np.linalg.eig(mat)[0]
            else:
                try:
                    r,q=(((-b/a)-p + (((b/a)+p)**2 - 4*((c/a)+(b/a)*p+(p**2)))**0.5)/2,((-b/a)-p - (((b/a)+p)**2 - 4*((c/a)+(b/a)*p+(p**2)))**0.5)/2)
                    return p,r,q
                except:
                    return np.linalg.eig(mat)[0]
        return quad_sol()   


                                                            #to get eigenvectors

def eigenvectors(mat):
    def char_matrix(mat):
        if mat.shape[1]==2:                
            if abs(mat[0][0])>abs(mat[0][1]):
                eigenvector=np.zeros(2)
                eigenvector[0]=1
                eigenvector[1]= mat[0][0]/mat[0][1]
            else:
                eigenvector=np.zeros(2)
                eigenvector[1]=1
                eigenvector[0]= mat[0][1]/mat[0][0]

        else: 
            eigenvector=np.zeros(3)
            if abs(mat[0][0])>=abs(mat[0][1]) and abs(mat[0][1])>abs(mat[0][2]):
                eigenvector[0]=1
                eigenvector[2]= (mat[1][0]-mat[0][0]*mat[1][1]/mat[0][1])/(mat[0][2]*mat[1][1]/mat[0][1]-mat[1][2])
                eigenvector[1]= -(mat[0][0]+mat[0][2]*eigenvector[2])/mat[0][1]

            elif abs(mat[0][1])>=abs(mat[0][0]) and abs(mat[0][0])>abs(mat[0][2]):
                eigenvector[1]=1
                eigenvector[2]= (mat[1][1]-mat[0][1]*mat[1][0]/mat[0][0])/(mat[0][2]*mat[1][0]/mat[0][0]-mat[1][2])
                eigenvector[0]= -(mat[0][1]+mat[0][2]*eigenvector[2])/mat[0][1]

            else:
                eigenvector[2]=1
                eigenvector[1]= (mat[1][2]-mat[0][2]*mat[1][0]/mat[0][0])/(mat[0][1]*mat[1][0]/mat[0][0]-mat[1][1])
                eigenvector[0]= -(mat[0][2]+mat[0][1]*eigenvector[2])/mat[0][1]

        return eigenvector
    if mat.shape[0] != mat.shape[1] or mat.shape[0]==1:
        return f'A matrix of order {order(mat)} has no eigenvector'
    vectors=[]

    for eigenvalue in eigenvalues(mat):
        char_mat=mat-np.identity(mat.shape[0])*eigenvalue
        vectors.append(char_matrix(char_mat))
    return vectors  

   

st.write("""
# Matrix
click "**operation button**" to perform some matrix operations or click the **Properties button**" to Get the rank, order, trace, determinant, cofactor, adjoint, inverse, eigenvalues, eigenvector, etc of your matrix**.
""")

Properties = st.button("Properties")
Operations = st.button("Operations")
matrix=fresh_prop()


r1,r2,r3,r4,r5='row0','row1','row2','row3','row4'
row_name=['row0','row1','row2','row3','row4']
rows = [r1,r2,r3,r4,r5]

st.write("**use space or alphabets to separate elements**")

for i in range(matrix.shape[0]):
   rows[i]= st.text_input(f'enter elements for {row_name[i]}.') 

new_matrix=matrix

mattt=[]
for row in range(len(matrix)):
    new_row=[]
    elvalue=[]
    num=len(rows[row])
    n=0
    for el in rows[row]:
        n+=1

        if el=='-' or el=='+':
            elvalue.append(el)
            continue
        else:
            try:
                a=float(el)
                elvalue.append(el)

            except:
                if elvalue==[] or elvalue==['-'] or elvalue==['+']:
                    continue
                else:
                    new_row.append(elvalue)
                    elvalue=[]
                    
        if n==num:
            new_row.append(elvalue)

    mattt.append(new_row)
    new_row=[]
    
    
    
for r in range(len(mattt)):
    realrow=[]
    for eleme in range(len(mattt[r])):
        reale=""
        for units in mattt[r][eleme]:
            reale+=units
        matrix[r][eleme]=float(reale)


display_mat(matrix)
det = st.button("Determinant")
if det:
    display_mat(determinant(matrix))

co= st.button("Cofactor")
if co:
    display_mat(cofactor(matrix))
   

trans = st.button("Transpose")
if trans:
    display_mat(transpose(matrix))
   
   

adje = st.button("Adjoint")
if adje:
    display_mat(adjoint(matrix))
   
   
inv = st.button("Inverse")
if inv:
    display_mat(inverse(matrix))
   
   
tr = st.button("Trace")
if tr:
    st.write(trace(matrix))

   

   
ra = st.button("Rank")
if ra:
    st.write(rank(matrix))
   
   
eva = st.button("Eigenvalues")
if eva:
    display_mat(eigenvalues(matrix))
   
   
eve = st.button("Eigenvectors")
if eve:
    st.write('Not yet ready')
   
info = st.button("Info")
if info:
   st.write('The trace of a matrix is also the sum of its eigenvalues.\n An eigenvalue multiplied to its respective eigenvector gives same vector as the cross product of the original vector and that eigenvector. ')
       
   

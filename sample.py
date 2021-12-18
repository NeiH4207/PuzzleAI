import numpy as np

# print random matrix mxn
def print_random_matrix(m, n):
    print("Random matrix:")
    s = set()
    for i in range(m):
        for j in range(n):
            s.add(i * n + j)
    
    arr = list(s)
    
    for i in range(m):
        for j in range(n):
            rd_num = np.random.randint(0, len(arr))
            print(arr[rd_num], end="\t")
            arr.pop(rd_num)
            
        print()
        
                    
    
        
print("Random matrix:")
print_random_matrix(3, 3)
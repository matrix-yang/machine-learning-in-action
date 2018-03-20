from numpy import *

mata=ones((3,4))  #eye zeros
print(mata)
print(mata.shape[0])
print(shape(mata))
print(mata.min(0))
matb=tile([0,2,3,4],(3,1))  #tile(1,(3.1)) 把1写3行，列数重复1次
print(matb)
print(mata-matb)

print(mata/(mata-matb)) #对应列相除

print(range(5))   #range(start, stop[, step])
for i in range(32):
    print(i)
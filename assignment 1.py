import random
def tanh(x):
    return(2/ (1+ pow(2.7182 ,-2*x))) -1
def random_weight():
    return random.uniform(-0.5, 0.5)
i1, i2= 0.05 , 0.10

w1,w2=random_weight(),random_weight()
w3,w4=random_weight(),random_weight()
w5,w6=random_weight(),random_weight()
w7,w8=random_weight(),random_weight()

b1,b2=0.5 , 0.7

net_H1=w1*i1+w2*i2+b1
net_H2=w3*i1+w4*i2+b1

output_H1=tanh(net_H1)
output_H2=tanh(net_H2)

net_OT1=w5*output_H1 + w6*output_H2 + b2
net_OT2=w7*output_H1 + w8*output_H2 + b2

output_OT1 =tanh(net_OT1)
output_OT2 =tanh(net_OT2)

print("OutPut :")
print("O1 =",output_OT1 )
print("O2 =",output_OT2 )
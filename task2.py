import random
import math

def tanh(x):
    return(math.exp(x) - math.exp(-x)) / (math.exp(x)  + math.exp(-x))
def tanh_derivative(x):
    return 1 - tanh(x) **2
def initial_weights(rows,cols):
    return[[random.uniform(0.5,0.5) for _ in range(cols)] for _ in range(rows)]
def dot_product(matrx,vector):
    return[sum(matrx[i][j]*vector[j] for j in range(len(vector))) for i in range(len(matrx))]

def forward_pass(inputs, wieghts_input_hidden , bias_hidden , wieghts_output_hidden , bias_output):
    hid_net=[dot_product(wieghts_input_hidden , inputs)[i] + bias_hidden[i] for i in range (len(bias_hidden))]
    hid_out=[tanh(h) for h in hid_net]

    out_net=[dot_product(wieghts_output_hidden , hid_out)[i] + bias_output[i] for i in range (len(bias_output))]
    out_out=[tanh(o) for o in out_net]
    return hid_out,out_out,hid_net,out_net

def backpropagition(inputs ,targets,hid_out , out_out , hid_net, out_net ,wieghts_input_hidden ,bias_hidden ,
                     wieghts_output_hidden , bias_output , lr):
    out_error=[targets[i] - out_out[i] for i in range(len(targets))]
    out_delta=[out_error[i] * tanh_derivative(out_net[i]) for i in range(len(out_net))]

    hiddden_error= [sum(wieghts_output_hidden[j][i] * out_delta[j] for j in range(len(out_delta))) for i in range(len(hid_out))]         
    hidden_delta=[hiddden_error[i] * tanh_derivative(hid_net[i]) for i in range(len(hid_net))]

    for i in range(len(wieghts_output_hidden)):
        for j in range(len(wieghts_output_hidden[i])):
            wieghts_output_hidden[i][j] += lr *out_delta[i] * hid_out[j]

    for i in range(len(bias_output)):
        bias_output[i] += lr* out_delta[i]
    
    for i in range(len(wieghts_input_hidden)):
        for j in range(len(wieghts_input_hidden[j])):
            wieghts_input_hidden[i][j]+=lr* hidden_delta[i]*inputs[j]
    
    for i in range(len(bias_hidden)):
        bias_hidden[i] += lr*hidden_delta[i]

random.seed(42)
wieghts_input_hidden=initial_weights(2,2)
wieghts_output_hidden=initial_weights(2,2)
bias_hidden=[0.5,0.5]
bias_output=[0.7,0.7]
inputs=[0.05,0.10]
targets=[0.1,0.99]
lr=0.5

hid_out,out_out,hid_net,out_net=forward_pass(inputs, wieghts_input_hidden , bias_hidden , wieghts_output_hidden , bias_output)
print(" OutPut before training :" , out_net)

backpropagition(inputs ,targets,hid_out , out_out , hid_net, out_net ,wieghts_input_hidden ,bias_hidden ,
                     wieghts_output_hidden , bias_output , lr)
hid_out,out_out,hid_net,out_net=forward_pass(inputs, wieghts_input_hidden , bias_hidden , wieghts_output_hidden , bias_output)
print(" OutPut After training :" , out_out)

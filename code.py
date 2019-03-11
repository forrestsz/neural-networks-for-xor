import numpy as np
import matplotlib.pyplot as plt

HIDDEN_LAYER_SIZE = 2
INPUT_LAYER= 2
NUM_LABELS = 1
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

def rand_initialize_weights(L_in, L_out, epsilon):
    epsilon_init = epsilon
    W = np.random.rand(L_out, 1 + L_in) *2*epsilon_init - epsilon_init
    return W

def sigmoid(x):
    return 1.0/ (1.0 + np.exp(-x))

def sigmoid_gradient(z):
    g = np.multiply(sigmoid(z), (1-sigmoid(z)))
    return g

def nn_cost_function(theta1, theta2, x, y):
    m = X.shape[0] #number of input
    D_1 = np.zeros(theta1.shape) #2*3
    D_2 = np.zeros(theta2.shape) #1*3
    h_total = np.zeros((m,1)) #output, m*1, probability
    for t in range(m): #loop for every simple
        a_1 = np.vstack((np.array([[1]]), X[t:t + 1, :].T))#including the bias layer, 3*1
        z_2 = np.dot(theta1, a_1) #2*1
        a_2 = np.vstack((np.array([[1]]),sigmoid(z_2))) #3*1
        z_3 = np.dot(theta2, a_2) #1*1
        a_3 = sigmoid(z_3) #1*1
        h = a_3
        h_total[t,:] = h
        delta_3 = h - y[t:t+1,:].T #1*1
        delta_2 = np.multiply(np.dot(theta2[:,1].T, delta_3),
                              sigmoid_gradient(z_2))
        D_2 = D_2 + np.dot(delta_3, a_2.T) #1*3
        D_1 = D_1 + np.dot(delta_2, a_1.T) #2*3
        #finished the loop of bp
    theta1_grad = (1.0 /m) * D_1 #2*3
    theta2_grad = (1.0 /m) * D_2 #1*3
    J = (1.0 /m) * np.sum(-y * np.log(h_total)-
                          (np.array([[1]])-y)*np.log(1-h_total))
    return {'theta1_grad': theta1_grad,
            'theta2_grad':theta2_grad,
            'J':J, 'h': h_total}

theta1 = rand_initialize_weights(INPUT_LAYER, HIDDEN_LAYER_SIZE, epsilon=1)
theta2 = rand_initialize_weights(HIDDEN_LAYER_SIZE, NUM_LABELS, epsilon=1)

iter_times = 10000
alpha = 0.5
result = {'J': [], 'h': []}
theta_s = {}

for i in range(iter_times):
    cost_fun_result = nn_cost_function(theta1 = theta1, theta2 = theta2, x=X, y=y)
    theta1_g = cost_fun_result.get('theta1_grad')
    theta2_g = cost_fun_result.get('theta2_grad')
    J = cost_fun_result.get('J')
    h_current = cost_fun_result.get('h') #4*1
    theta1 -= alpha * theta1_g
    theta2 -= alpha * theta2_g
    result['J'].append(J)
    result['h'].append(h_current) #[[0,1,1,0],[0,1,1,0],...[0,1,1,0]]
    if i==0: #or i==(iter_times-1):
        print('The first times theta1:',theta1)
        print('The first times theta2',theta2)
        theta_s['theta1_'+str(i)] = theta1.copy()
        theta_s['theta2_'+str(i)] = theta2.copy()
        print('================================================')
    elif i==(iter_times-1):
        print('The last times theta1:',theta1)
        print('The last times theta2',theta2)
        theta_s['theta1_' + str(i)] = theta1.copy()
        theta_s['theta2_' + str(i)] = theta2.copy()

plt.plot(result.get('J'))
plt.show()
print('================================================')
print('The first times result:',result.get('h')[0])
print('The last times result:',result.get('h')[-1])


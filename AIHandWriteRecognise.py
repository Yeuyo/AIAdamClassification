import numpy
from mlxtend.data import loadlocal_mnist

class AdamOptimizer():
    def __init__(self, alp=-0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alp = alp
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
    def update(self, t, m_dw, v_dw, w, dw):
        m_dw = self.beta1*m_dw + (1-self.beta1)*dw # https://arxiv.org/pdf/1412.6980.pdf
        v_dw = self.beta2*v_dw + (1-self.beta2)*(dw**2)
        m_dw_corr = m_dw/(1-self.beta1**t)
        v_dw_corr = v_dw/(1-self.beta2**t)
        w = w - self.alp*(m_dw_corr/(numpy.sqrt(v_dw_corr)+self.epsilon))
        return m_dw, v_dw, w

def randInitializeWeights(L_in, L_out):
    # Randomly initialize the weights of a layer with L_in
    # incoming connections and L_out outgoing connections
    # L_in is added one to account for the bias node
    epsilon_init = 0.12
    W = numpy.random.random((1 + L_in, L_out)) * 2 * epsilon_init - epsilon_init
    return W

def sigmoid(z):
    # Compute Sigmoid function
    z = numpy.reshape(z,[len(z),1])
    numpy.warnings.filterwarnings('ignore', 'overflow')
    g = 1.0 / (1.0 + numpy.exp(-z))
    g = numpy.append(1, g) # To account for the bias node
    return g

def applyFeedForward(X, w0):
    netoutput = [i for i in range(num_of_layers+1)]
    netinput = [i for i in range(num_of_layers+1)]
    netoutput[0] = numpy.append(1, X) # Add in the bias node
    for i in range(num_of_layers):
        netinput[i+1] = numpy.matmul(numpy.transpose(w0[i]), netoutput[i])
        netoutput[i+1] = sigmoid(netinput[i+1])    
    netoutput[num_of_layers] = netoutput[num_of_layers][1:] # Output do not have bias
    return netoutput

def costFunction():
    # Compute cost (also known as loss or error)
    cost = 0
    for i in range(sample_size):
        nodes = applyFeedForward(X[i], w0)
        predict = nodes[num_of_layers]
        actual = y[i]
        cost = cost + sum(pow((actual - predict), 2)/2)

    cost = cost / sample_size
    return cost

# Training sets
X, y0 = loadlocal_mnist(images_path='train-images.idx3-ubyte',
                       labels_path='train-labels.idx1-ubyte')
y = numpy.zeros([X.shape[0],10])
for i in range(X.shape[0]):
    y[i,y0[i]]=1
X=X[:10,:] # If you only want to use some of the training sets
y=y[:10,:]


# Testing sets
X1, y1 = loadlocal_mnist(images_path='t10k-images.idx3-ubyte',
                         labels_path='t10k-labels.idx1-ubyte')
y01 = numpy.zeros([X1.shape[0],10])
for i in range(X1.shape[0]):
    y01[i,y1[i]]=1
X1=X1[:2,:]   # If you only want to use some of the test sets
y01=y01[:2,:]

# Setup the parameters
criterion = 1e-9
MaxIter = 1000
hidden_layer_size = [5, 5]                 # 2 hidden layers with 5 nodes each
input_layer_size  = X.shape[1]             # This should be the number of pixels in the image
sample_size = X.shape[0]                   # Number of samples given (numbers of known solution)
num_labels = y.shape[1]                    # This should be the number of classification/labels  
                                           # (with the first column meaning the solution is '1'
                                           # and so on)
test_size = X1.shape[0]
num_of_layers = len(hidden_layer_size) + 1 # plus input layer

w0 = [0.0 for i in range(num_of_layers)]
w0[0] = randInitializeWeights(input_layer_size, hidden_layer_size[0])
if len(hidden_layer_size) > 1:
    for i in range(len(hidden_layer_size) - 1):
        w0[i+1] = randInitializeWeights(hidden_layer_size[i], hidden_layer_size[i+1])
w0[num_of_layers-1] = randInitializeWeights(hidden_layer_size[len(hidden_layer_size) - 1], num_labels)
adam = AdamOptimizer()
t = 1 
converged = False

# Adam's optimization parameters
m_dw = [0.0 for i in range(num_of_layers+1)] # +1 for output layer
v_dw = [0.0 for i in range(num_of_layers+1)]

# Program starting
for t in range(MaxIter):
    for i in range(sample_size):
        nodes = applyFeedForward(X[i], w0)
        predict = nodes[num_of_layers]
        actual = y[i]
        sigmas = [i for i in range(num_of_layers+1)] # Input layer will not have error
        sigmas[num_of_layers] = actual - predict
        for j in range(num_of_layers - 1, -1, -1):
            if sigmas[j + 1].shape[0] == 1:
                sigmas[j] = w0[j] * sigmas[j + 1]
            else:
                if j == num_of_layers - 1: # Output layer do not have bias
                    sigmas[j] = numpy.matmul(w0[j], sigmas[j + 1])
                else:
                    sigmas[j] = numpy.matmul(w0[j], sigmas[j + 1][1:])
        numpy.warnings.filterwarnings('ignore', category=numpy.VisibleDeprecationWarning) 
        derivative_of_sigmoid = nodes * (numpy.array([1]) - nodes)
        sigmas = derivative_of_sigmoid * sigmas
        for j in range(num_of_layers):
            if j == num_of_layers - 1:
                dw = numpy.reshape(nodes[j],[len(nodes[j]),1]) * numpy.transpose(numpy.reshape(sigmas[j+1],[len(sigmas[j+1]),1])) # nodes[j] * numpy.transpose(sigmas[j+1]) # Output layer do not have bias to remove
            else:
                dw = numpy.reshape(nodes[j],[len(nodes[j]),1]) * numpy.transpose(numpy.reshape(sigmas[j+1][1:],[len(sigmas[j+1][1:]),1])) # nodes[j] * numpy.transpose(sigmas[j+1][1:])
            m_dw[j], v_dw[j], w0[j] = adam.update(t=t+1, m_dw=m_dw[j], v_dw=v_dw[j], w=w0[j], dw=dw)

for i in range(test_size):
    nodes = applyFeedForward(X1[i], w0)
    predict = nodes[num_of_layers]
    actual = y01[i]
    # valuepredict = numpy.zeros_like(predict)
    # valuepredict[numpy.arange(len(predict)),predict.argmax(1)] = 1
    print(numpy.transpose(X1[i])," -> actual: ", numpy.transpose(actual),", predict: ", numpy.transpose(predict))

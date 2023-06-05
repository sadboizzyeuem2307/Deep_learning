from libary import *

class NeuralNetwork:
   def __init__(self, layers, learning_rate):
      self.layers = layers
      self.learning_rate = learning_rate

      self.b = []
      self.W = []
      
      # Create unit

      for i in range(0, len(layers) - 1):
         w_ = np.random.randn(layers[i], layers[i + 1])
         b_ = np.zeros((layers[i + 1], 1))

         self.W.append(w_/np.sqrt(layers[i]))
         self.b.append(b_)
      
   def fit_train(self, x, y):
      A = [x]

      # fordward
      out_put = A[-1]

      for i in range(0 , len(self.layers) - 1):
         out_put = sigmoid(np.dot(out_put, self.W[i]) + self.b[i].T)
         A.append(out_put)
      
      # backpropagation
      dL = [-(y / (A[-1]) - (1 - y) /(1 - A[-1]))]
      # y = out_put real, A[-1] out_put of fordward
      dW = []
      db = []

      for i in reversed(range(0, len(self.layers) - 1)):
         dw_ = np.dot((A[i]).T, dL[-1] * derivative_s(A[i + 1]))
         db_ = (np.sum(dL[-1] * derivative_s(A[i + 1]), axis = 0)).reshape(-1,1)
         dl_ = np.dot(dL[-1] * derivative_s(A[i + 1]), self.W[i].T)
         dL.append(dl_)
         dW.append(dw_)
         db.append(db_)
      
      dW = dW[::-1]
      db = db[::-1]
      # gradient descent

      for i in range(0, len(self.layers) - 1):
         self.W[i] = self.W[i] - self.learning_rate * dW[i]
         self.b[i] = self.b[i] - self.learning_rate * db[i]
      
   def fit(self, x, y, epochs, verbose):
      for epoch in range(0, epochs):
         self.fit_train(x,y)
         if epoch % verbose == 0:
            loss = self.c_loss(x, y)
            print("Epochs {}, loss {}".format(epoch, loss))

   def predict(self, x):
      for i in range(0, len(self.layers) - 1):
         x = sigmoid(np.dot(x, self.W[i]) + (self.b[i].T))
      return x
   
   def c_loss(self, X, y):
      y_predict = self.predict(X)
      # return np.sum((y_predict-y)**2)/2
      return -(np.sum(y*np.log(y_predict) + (1-y)*np.log(1-y_predict)))
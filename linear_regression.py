import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def normalize(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    me = np.ones(arr.shape)*mean
    return (arr-me)/std
class Linear_regression:
    def __init__(self,x,y,iter,lr,threshold):
        self.m0 = 0
        self.m1 = 0
        n = len(x)
        self.cost = []
        self.m0_arr = []
        self.m1_arr = []
        for i in range(iter):
            arr = np.array([self.m0,self.m1]).reshape(-1,1)
            y_predict = np.dot(x,arr)[:,0] 
            current_cost = self.mean_squared(y,y_predict,n)
            self.cost.append(current_cost)

            d_m1 = -(2/n) * sum(x[:,1]*(y-y_predict)) 
            d_m0 = -(2/n) * sum(y-y_predict)          

            self.m0 = self.m0 - lr*d_m0
            self.m1 = self.m1 - lr*d_m1
            self.m0_arr.append(self.m0)                
            self.m1_arr.append(self.m1)

            print(f"Iteration {i+1}: Cost = {current_cost}. Phuong trinh: y = {self.m1}x+{self.m0}")
            
            if (abs(d_m1) <= threshold) and (abs(d_m0)<=threshold):
                break                                   
        
    def coefficent(self):
        return np.array([self.m0,self.m1])
    
    def predict(self,x):
        return (self.m0 + self.m1 * x)
    
    def plot_cost(self):
        fig,axs = plt.subplots(2)
        axs[0].plot(self.m0_arr,self.cost)
        axs[0].set_title("Cost and m0")
        axs[0].set_xlabel("m0")
        axs[0].set_ylabel("Cost")
        axs[1].plot(self.m1_arr,self.cost)
        axs[1].set_title("Cost and m1")
        axs[1].set_xlabel("m1")
        axs[1].set_ylabel("Cost")
        plt.show()
    
    def mean_squared(self,y,y_predict,n):
        return 1/(2*n)*sum((y-y_predict)**2)

           
if __name__ == "__main__":
#Read file data 
    file = "D:/fun_ML/sample.csv"
    file_read = pd.read_csv(file)
    data = np.array(file_read)
    y = data[:,0]
    x = data[:,1]
    x1 = x.reshape(-1,1)
    a = 2*np.ones(x1.shape)
    x1 = np.stack((a,x1),axis=1)[:,:,0]

#Using LinearRegression class in sklearn library:
    reg = LinearRegression().fit(x1,y)
    y_pred_tool = reg.predict(x1)
    print('MSE using sklearn: ',mean_squared_error(y,y_pred_tool))
    fig, axs = plt.subplots(2)
    axs[0].scatter(x,y,color='b')
    axs[0].plot(x,y_pred_tool,color='k')
    axs[0].set_title('Using sklearn Linear Regression')

# Using own class:
    linear = Linear_regression(x1,y,1000,1e-9,10)
    y_pred = linear.predict(x)
    axs[1].set_title('Using own Linear Regression')
    axs[1].scatter(x,y,color='b')
    axs[1].plot(x,y_pred,color='k')
    
#Vẽ đồ thị và kiểm tra hàm loss:
    plt.show()
    linear.plot_cost()







    
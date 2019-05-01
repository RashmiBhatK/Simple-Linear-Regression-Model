#The optimal values of m and b can be actually calculated with way less effort than doing a linear regression. 
#this is just to demonstrate gradient descent

from numpy import *
import matplotlib.pyplot as plt

# y = mx + b
# m is slope, b is y-intercept
def compute_error(b, m, points):
    totalError = 0
    for i in range(0, len(points) - 30):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points) - 30)

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points) - 30)
    for i in range(0, len(points) - 30):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

def run():
    points = genfromtxt("data.csv", delimiter=",")
    lr = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_weight = 0 # initial slope guess
    numofiterations = 1000
    print( "At Start: gradient descent at bias = %-.2f, weight = %-.2f, error = %-.2f" %(initial_b, initial_weight, compute_error(initial_b, initial_weight, points)))
    print("Running, ")
    [b, weight] = gradient_descent_runner(points, initial_b, initial_weight, lr, numofiterations)
    print("After %d iterations: bias = %-.2f, weight = %-.2f, error = %-.2f" %(numofiterations, b, weight, compute_error(b, weight, points)))

    # Visualizations of training set
    plt.scatter([points[i, 0] for i in range(0, 70)], [points[i, 1] for i in range(0, 70)], color = 'red')
    plt.plot([points[i, 0] for i in range(0, 70)], [(weight * x + b) for x in  [points[i,0] for i in range(0, 70)]], color = 'blue')
    plt.title('Test Scores vs No. of hours of study (Training set)')
    plt.xlabel('No. of hours of study')
    plt.ylabel('Test Scores')
    plt.show()

    # Visualizations of testing set
    plt.scatter([points[i, 0] for i in range(70, len(points))], [points[i, 1] for i in range(70, len(points))], color = 'red')
    plt.plot([points[i, 0] for i in range(0, 70)], [(weight * x + b) for x in  [points[i,0] for i in range(0, 70)]], color = 'blue')
    plt.title('Test Scores vs No. of hours of study (Testing set)')
    plt.xlabel('No. of hours of study')
    plt.ylabel('Test Scores')
    plt.show()

    predicted_y = [(weight * x + b) for x in [points[i,0] for i in range(70, len(points))]]
    actual_y = [points[i, 1] for i in range(70, len(points))]

    accurate_count = 0

    for i in range(30):
        error = abs(actual_y[i] - predicted_y[i])
        error_percent = error / actual_y[i]
        if(error_percent < 0.1):
            accurate_count += 1

    print("Accuracy : %-.2f" %(accurate_count / 30))        

if __name__ == '__main__':
    run()             
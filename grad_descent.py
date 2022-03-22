import numpy as np
import numpy.linalg as LA


def init_sep_vector():
    label_vec = np.ones(100)
    value_vec = np.ones(100)
    for i in range(50):
        label_vec[i] = -1
    mid = 50
    for i in range(50):
        value_vec[i] = -1 * mid
        value_vec[99 - i] = mid
        mid -= 1
    return value_vec, label_vec


def corrupt(in_x, in_y):
    for i in range(5):
        in_y[i] *= -1
        in_y[99 - i] *= -1
    return in_x, in_y


def grad_descent(in_x, in_y, w):
    learning_rate = 0.01
    # for 100 iters
    for i in range(100):
        total_grad = 0
        for index in range(100):
            total_grad += gradient(in_x[index], in_y[index], w)
        w += -1 * learning_rate * (total_grad / 100)
    return w


def gradient(x_point, y_point, w):
    grad = -y_point * x_point
    denom = 1 + np.exp(y_point * w * x_point)
    return grad / denom


def closed_lsr(matrix_a, vector_b):
    return LA.inv(matrix_a.T @ matrix_a) @ matrix_a.T @ vector_b


def setup_least_squares():
    n = 500
    m = 2 * n
    matrix_a = np.matrix([[np.random.uniform(-1, 1) for i in range(n)] for j in range(m)])
    hidden_x = np.array([np.random.uniform(-1, 1) for i in range(n)])
    eta = np.array([np.random.normal(0, np.sqrt(0.5)) for i in range(m)])
    vector_b = np.array(np.dot(matrix_a, hidden_x.T) + eta)[0]
    return matrix_a, hidden_x, vector_b


def grad_least_squares(in_a, in_b, in_hidden_x):
    our_x = np.matrix(np.zeros(len(in_hidden_x))).T
    lr = 0.1
    for i in range(50):
        our_x -= lr*2*np.dot(in_a.T, (np.dot(in_a, in_hidden_x.T) - in_b).T)
    return our_x


def euc_distance(expected, produced):
    # expected = np.squeeze(expected)
    # print(expected.shape)
    # produced = produced.reshape((-1,))
    # print(produced.shape)
    return LA.norm(expected, produced.all())


def grad(x_val, y_val, index, n):
    if index <= n:
        ai = index/n
        bi = -1
    else:
        ai = (index - n) / n
        bi = 1

    return 2*(x_val-ai), 2*(y_val-bi)


def sgd(in_x, in_y, num_iters=200, mode="standard"):
    lr = 0.1
    a_learned, b_learned = 0, 0
    for i in range(num_iters):
        if mode == "frac":
            lr = 0.1/(i+1)
        if mode == "root":
            lr = 0.1/(np.sqrt(i)+1)

        stoch_choice = np.random.randint(0, len(in_x))
        a_diff, b_diff = grad(in_x[stoch_choice], in_y[stoch_choice], stoch_choice, len(in_x))
        a_learned -= lr * a_diff
        b_learned -= lr * b_diff

    return a_learned, b_learned


if __name__ == "__main__":
    # problem 1c
    x, y = init_sep_vector()
    print(f'Result of gradient descent in 1D: {grad_descent(x, y, -1)}')

    # problem 1d.
    x, y = corrupt(x, y)
    print(f'Corrupted gradient descent: {grad_descent(x, y, -1)}')

    # problem 3c.
    a, x, b = setup_least_squares()

    # problem 3d.
    print(f'Euclidean distance from hidden x: {euc_distance(x, grad_least_squares(a, b, x))}')

    # problem 3e.
    print(f'Euclidean distance from hidden x w/ inverse: {euc_distance(x, closed_lsr(a, b))}')

    # problem 4c.
    x, y = init_sep_vector()
    x, y = corrupt(x, y)
    print(f'SGD: {sgd(x, y)}')
    print(f'SGD w/ lr=0.1/t+1 : {sgd(x, y, mode="frac")}')
    print(f'SGD w/ lr=0.1/sqrt(t)+1 : {sgd(x, y, mode="root")}')

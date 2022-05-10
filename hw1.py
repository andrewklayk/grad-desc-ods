import numpy as np
import matplotlib.pyplot as plt
import time


def generate_points(
    loc_1: tuple[float,float] = (0,0),
    loc_2: tuple[float,float] = (6,6),
    prop_of_labels: float = 0.01,
    n: int = 1000):
    # generate two clusters of points
    n_labeled = int(n*prop_of_labels)
    cluster_1 = np.array(
        [np.random.normal(loc=loc_1[0], size=n//2),
        np.random.normal(loc=loc_1[1], size=n//2)]).T.tolist()
    cluster_2 = np.array(
        [np.random.normal(loc=loc_2[0], size=n//2),
        np.random.normal(loc=loc_2[1], size=n//2)]).T.tolist()
    # divide into labeled and unlabeled
    c1l = cluster_1[:n_labeled//2]
    c2l = cluster_2[:n_labeled//2]
    x_l = c1l + c2l
    x_u = cluster_1[n_labeled//2:] + cluster_2[n_labeled//2:]
    # generate labels
    labels = [1. for _ in c1l] + [-1. for _ in c2l]
    # true labels for unlabeled points
    u_labels = [1. for _ in cluster_1[n_labeled//2:]] + [-1. for _ in cluster_2[n_labeled//2:]]
    return x_l, x_u, labels, u_labels

def calc_similarity(x_l, x_u, save_to=None):
    """
    calculate two similarity matrices:
    for sim. between labeled and unlabeled,
    and for sim. between unlabeled and unlabeled
    """
    sim_l = np.zeros((np.shape(x_l)[0], np.shape(x_u)[0]))
    sim_u = np.zeros((np.shape(x_u)[0], np.shape(x_u)[0]))

    for i, xi in enumerate(x_l):
        for j, xj in enumerate(x_u):
            if not (i == j):
                sim_l[i,j] = similarity_labeled(xi,xj)
    for i, xi in enumerate(x_u):
        for j, xj in enumerate(x_u):
            if not (i == j):
                sim_u[i,j] = similarity_unlabeled(xi,xj)

    sim_l /= np.max(sim_l)
    sim_u /= np.max(sim_u)

    if save_to:
        np.savetxt(save_to[0], sim_l)
        np.savetxt(save_to[1], sim_u)

    return sim_l, sim_u

def similarity_labeled(point_1, point_2):
    return 5/(np.sqrt((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2)+0.0001)
    
def similarity_unlabeled(point_1, point_2):
    return similarity_labeled(point_1, point_2)/5

def fx(y_l, y_u, sim_l, sim_u):
    u = np.dot(np.array(sim_l).flatten(), np.square(np.subtract.outer(y_l, y_u)).T.flatten())
    l = 0.5*np.dot(np.array(sim_u).flatten(), np.square(np.subtract.outer(y_u, y_u)).T.flatten())
    return u + l

def gradient_partial(
    y_l: list,
    y_u: list,
    y_j: float,
    sim_l: list,
    sim_u: list,
    j: int = None) -> float:
    """
    Calculates the partial gradient with respect to the j-th label
    """
    res = 0
    # first term of the partial gradient
    res += np.sum([sim_l[i,j]*(y_j-y_i) for i,y_i in enumerate(y_l)])
    # second term of the partial gradient
    res += np.sum([sim_u[i,j]*(y_j-y_i) for i,y_i in enumerate(y_u)])
    return res*2

def gradient(
    y_l: list, 
    y_u: list, 
    sim_l, 
    sim_u) -> float:
    """
    Calculates the full gradient (vector of all partial gradients)
    """
    return np.array([gradient_partial(y_l=y_l, y_u=y_u, y_j=y_j, sim_l=sim_l, sim_u = sim_u, j=j) for j, y_j in enumerate(y_u)])

def gd(
    y_l: list,
    y_u: list[float],
    sim_l:np.ndarray,
    sim_u:np.ndarray,
    step_size: float = 0.1,
    tol: float = 1,
    max_iters: int = 1000,
    method: str = 'gd'):
    """
    Full gradient descent algorithm.
    :param y_l: labeled examples labels
    :param y_u: unlabeled examples labels initialization (usually random {-1,1} or 0)
    :param sim_l: labeled-labeled examples similarity matrix
    :param sim_u: labeled-unlabeled examples similarity matrix
    :param step_size: float for a fixed step size, TODO line search
    :param tol: tolerance value for stopping condition
    :param max_iters: max number of iterations
    :param method: 'gd' for gradient descent, 'bcgdc' for cyclic bcgd, 'bcgdr' for randomized bcgd; block size 1 in both cases
    """
    # save target function values over iterations
    y_history = [y_u]
    # save time at iteration end
    time_history = []
    if not method == 'gd':
        max_iters *= 1000 
    start_time = time.process_time()
    for k in range(max_iters):
        # normal gradient descent
        if method == 'gd':
            # TODO: try replacing with optimized (from formula) variant
            grad_k = gradient(y_l=y_l, y_u=y_history[-1], sim_l=sim_l, sim_u=sim_u)
            y_new = y_history[-1] - step_size*grad_k

        # bcgd with random blocks of size 1
        elif method == 'bcgdr':
            # Calculate the full gradient one time in the beginning
            # Each next iteration updates one el-t of the gradient
            if k == 0:
                grad_k = gradient(y_l=y_l, y_u=y_history[-1], sim_l=sim_l, sim_u=sim_u)
            # Choose the el-t to update
            j = np.random.randint(len(y_u))
            # Calculate the partial gradient
            grad_k[j] = gradient_partial(y_l, y_history[-1], y_history[-1][j], sim_l, sim_u, j)
            y_new = y_history[-1].copy()
            y_new[j] = y_history[-1][j] - step_size*grad_k[j]

        # bcgd with cyclic blocks of size 1
        elif method == 'bcgdc':
            # Calculate the full gradient one time in the beginning
            # Each next iteration updates one el-t of the gradient
            if k == 0:
                grad_k = gradient(y_l=y_l, y_u=y_history[-1], sim_l=sim_l, sim_u=sim_u)
            # Choose the el-t to update - loops from 0 to len(y_u)-1
            j = k % len(y_u)
            # Calculate the partial gradient
            grad_k[j] = gradient_partial(y_l, y_history[-1], y_history[-1][j], sim_l, sim_u, j)
            y_new = y_history[-1].copy()
            y_new[j] = y_history[-1][j] - step_size*grad_k[j]

        time_history.append(time.process_time())

        # stopping condition
        if np.linalg.norm(grad_k) < tol:
            print('\nTERMINATED BY CONDITION\n on iter ' + str(k) + '\n')
            return y_history, np.array(time_history) - start_time
        y_history.append(y_new)
    print('TERMINATED BY MAX ITER\n')
    return y_history, np.array(time_history) - start_time


def main():
    load_x = True
    save = True

    # Choose method
    m = 'bcgdc'

    # initialize points
    if load_x:
        x_l = np.loadtxt('x_l.txt')
        x_u = np.loadtxt('x_u.txt')
        sim_l = np.loadtxt('sim_l.txt')
        sim_u = np.loadtxt('sim_u.txt')
        labels = np.loadtxt('labels.txt')
        u_labels = np.loadtxt('u_labels.txt')
        start_labels = np.loadtxt('start_labels.txt')
    else:
        # generate points and calculate similarity matrices
        x_l, x_u, labels, u_labels = generate_points()
        sim_l, sim_u = calc_similarity(x_l, x_u)
        # Initialize unknown labels with either {-1,1} or all 0
        start_labels = [np.random.choice(tuple({-1,1})) for _ in x_u]
        if save:
            np.savetxt('x_l.txt', x_l)
            np.savetxt('x_u.txt', x_u)
            np.savetxt('labels.txt', labels)
            np.savetxt('u_labels.txt', u_labels)
            np.savetxt('sim_l.txt', sim_l)
            np.savetxt('sim_u.txt', sim_u)
            np.savetxt('start_labels.txt', start_labels)

    # do gradient descent
    y_hist, time_hist = gd(
        y_l=labels, 
        y_u=start_labels,
        sim_l=sim_l, 
        sim_u=sim_u, 
        method=m)
    guess_labels = [[-1 if y < 0 else 1 for y in y_k] for y_k in y_hist]
    acc_hist = np.array([np.sum(np.array(gl) == np.array(u_labels)) for gl in guess_labels])/len(y_hist[-1])
    # save results to file
    if save:
        np.savetxt('y_hist_{0}.txt'.format(m), y_hist)
        np.savetxt('acc_hist_{0}.txt'.format(m),acc_hist)
        np.savetxt('time_hist_{0}.txt'.format(m),time_hist)

    _, ax1 = plt.subplots(nrows=1)
    ax1.plot(time_hist, acc_hist)
    ax1.set_ylabel('Accuracy')
    print(acc_hist[-1])

    # The commented code below plots iterations vs. function value
    # It takes a REALLY long time to calculate f(x) for each y in y_hist

    # ax1.text(len(time_hist)/4, 0.5, "End function value: {0}\n Iterations done: {1}\n End accuracy: {2}".format(
    #     fx(labels, y_hist[-1], sim_l, sim_u), 
    #     len(y_hist),
    #     acc_hist[-1]), bbox=dict(facecolor='red', alpha=0.5))

    # if load_res:
    #     f_hist = np.loadtxt('f_hist.txt')
    # else:
    #     f_hist = [fx(labels, yk, sim_l, sim_u) for yk in y_hist]    
    #     if save:
    #         np.savetxt('f_hist_{0}.txt'.format(m), f_hist)

    # ax2.plot(np.arange(len(y_hist))/1000, f_hist)
    # ax2.set_ylabel('Target f-n value')
    # ax2.set_xlabel('N of iterations')
    plt.savefig('time_{0}.png'.format(m))
    #plt.show()


if __name__ == "__main__":
    main()
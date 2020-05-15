import numpy as np
import matplotlib.pyplot as plt

A = np.genfromtxt('data/flame.csv')
testdata = A.copy()[:,1:].astype('float')
N = 4

def k_means(data, clusters):
    dim = data.shape[1]
    clst = np.zeros([3, 2])
    avg = np.mean(data, axis = 0)
    sd = np.std(data, axis = 0)
    clst = avg + np.random.randn(clusters, dim) * sd
    d  = np.zeros([data.shape[0], clusters])
    a = 0
    change = 1
    while change != 0:
        a += 1
        for i in range(clusters):
            d[:,i] = np.linalg.norm(data - clst[i], axis = 1)
        y = np.argmin(d, axis = 1)
        change = clst.copy()
        for i in range(clusters):
            clst[i] = np.mean(data[y == i], axis = 0)
        change = np.linalg.norm(clst - change)
        # for j in range(clusters):
        #     plt.plot(data[y == j][:,0], data[y == j][:,1], '.')
        # plt.plot(clst[:,0], clst[:,1], 'D', color='red')
        # plt.title('Iteration no.:'+str(a))
        # plt.show()
    return clst, y, a

k_m = k_means(testdata, N)
skr = k_m[0]


if N == 2:
    a = 0
    for i in range(testdata.shape[0]):
        a += np.array_equal(k_m[1][i], A[i,0])
    prob = a/testdata.shape[0] # In case C1 is C0 and C0 is C1
    if prob < .5:
        prob = 1 - prob
    print('Accuracy:', prob)


plt.subplot(121)
plt.plot(testdata[A[:,0] == 0][:,0], testdata[A[:,0] == 0][:,1], '.')
plt.plot(testdata[A[:,0] == 1][:,0], testdata[A[:,0] == 1][:,1], '.')
plt.plot(skr[:,0], skr[:,1], 'D', color='red')
plt.title('The actual classes')

plt.subplot(122)
for j in range(N):
    plt.plot(testdata[k_m[1] == j][:,0], testdata[k_m[1] == j][:,1], '.')
plt.plot(skr[:,0], skr[:,1], 'D', color='red')
plt.title('K-means classes')

plt.show()

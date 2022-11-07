from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os

cluster_count = 16
eps = 1e-6

def k_means(image, k):

    random_h = np.random.choice(image.shape[0], k, replace=False)
    random_w = np.random.choice(image.shape[1], k, replace=False)
    mu = np.array([image[random_h[i], random_w[i]] for i in range(k)]).astype('int')
    it = 0

    while True:
        it += 1
        print(f'Iteration: {it}')
        clusters = np.array([np.argmin(np.linalg.norm(image[h, w] - mu, axis=1))
                             for h in range(image.shape[0]) for w in range(image.shape[1])])

        clusters = np.reshape(clusters, (image.shape[0], image.shape[1]))
        for j in range(k):
            group = image[clusters == j]
            if group.shape[0] > 0:
                mu[j] = np.mean(group, axis=0)

        if it > 50:
            break

    return mu.astype(int)

def main():

    small_image = imread('../data/peppers-small.tiff')
    mu = k_means(small_image, cluster_count)

    big_image = imread('../data/peppers-large.tiff')

    test = np.copy(big_image)

    for i in range(big_image.shape[0]):
        for j in range(big_image.shape[1]):
            test[i, j] = mu[np.argmin(np.linalg.norm(mu - big_image[i, j], axis=1))]

    plt.figure(1)
    plt.axis('off')
    plt.title("Updated Image")
    plt.imshow(test);

    print(test.shape)
    savepath = os.path.join('../output', 'updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()

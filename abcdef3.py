import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn import decomposition

n_components = 10
image_shape = (8, 8)

digits = load_digits()
digits = digits.data
print(digits.shape)

n_samples, n_features = digits.shape
estimator = decomposition.PCA(n_components=n_components, svd_solver='randomized', whiten=True)
digits_recons = estimator.inverse_transform(estimator.fit_transform(digits))

# show 5 randomly chosen digits and their PCA reconstructions with 10 dominant eigenvectors
indices = np.random.choice(n_samples, 5, replace=False)
plt.figure(figsize=(5,2))
for i in range(len(indices)):
    plt.subplot(1,5,i+1), plt.imshow(np.reshape(digits[indices[i],:], image_shape)), plt.axis('off')
plt.suptitle('Original', size=25)
plt.show()
plt.figure(figsize=(5,2))
for i in range(len(indices)):
    plt.subplot(1,5,i+1), plt.imshow(np.reshape(digits_recons[indices[i],:], image_shape)), plt.axis('off')
plt.suptitle('PCA reconstructed'.format(n_components), size=25)
plt.show()
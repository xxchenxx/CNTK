from resnet import ResNet18
import neural_tangents as nt
from utils import load_cifar
import numpy as np
import sys
from jax import random
key1, key2 = random.split(random.PRNGKey(int(sys.argv[1])), 2)
init_fn, apply_fn = ResNet18(num_classes=10, img_dim=(32, 32))
(X_train, y_train), (X_test, y_test) = load_cifar()
X = np.concatenate((X_train, X_test), axis = 0)
N = X.shape[0]
N_train = X_train.shape[0]
N_test = X_test.shape[0]

kernel_fn = nt.empirical_kernel_fn(apply_fn, vmap_axes=0, implementation=2)
_, params = init_fn(key1, X_train.shape)
k_test_train = kernel_fn(X_test, X_train, params)
k_train_train = kernel_fn(X_train, None, params)

from jax.example_libraries import stax
from neural_tangents import predict
cross_entropy = lambda fx, y_hat: -np.mean(stax.logsoftmax(fx) * y_hat)
predict_fn = predict.gradient_descent(cross_entropy, k_test_train,
                                      y_train, 0.01, 0.9)

fx_train_0 = apply_fn(params, X_train)
fx_test_0 = apply_fn(params, X_test)

fx_train_t, fx_test_t = predict_fn(1e-7, fx_train_0, fx_test_0,
                                   k_test_train)
import pickle
pickle.dump([fx_train_t, fx_test_t], open(f"{sys.argv[1]}.pkl", "wb"))
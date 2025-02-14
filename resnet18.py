from resnet import ResNet18
import neural_tangents as nt
from utils import load_cifar
import numpy as np
import sys
import jax
from jax import random
key1, key2 = random.split(random.PRNGKey(int(sys.argv[1])), 2)
init_fn, apply_fn = ResNet18(num_classes=10, img_dim=(32, 32))
(X_train, y_train), (X_test, y_test) = load_cifar()
X_train = jax.numpy.array(X_train)
X_test = jax.numpy.array(X_test)
y_train = jax.numpy.array(y_train)
y_test = jax.numpy.array(y_test)


print(X_train.shape)
_, params = init_fn(key1, X_train.shape)
kernel_fn = nt.empirical_kernel_fn(apply_fn, vmap_axes=0, implementation=2)
kernel_fn_batched = nt.batch(kernel_fn, device_count=-1, batch_size=5)

k_test_train = kernel_fn_batched(X_test, X_train, 'ntk',params)
k_train_train = kernel_fn_batched(X_train, None, 'ntk', params)

from jax.example_libraries import stax
from neural_tangents import predict
cross_entropy = lambda fx, y_hat: -np.mean(stax.logsoftmax(fx) * y_hat)
predict_fn = predict.gradient_descent(cross_entropy, k_train_train,
                                      y_train, 0.01, 0.9)

fx_train_0 = apply_fn(params, X_train)
fx_test_0 = apply_fn(params, X_test)

fx_train_t, fx_test_t = predict_fn(1e-7, fx_train_0, fx_test_0,
                                   k_test_train)
import pickle
pickle.dump([fx_train_t, fx_test_t], open(f"{sys.argv[1]}.pkl", "wb"))
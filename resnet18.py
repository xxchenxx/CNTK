from resnet import ResNet18
import neural_tangents as nt
from utils import load_cifar
import numpy as np
init_fn, apply_fn, kernel_fn = ResNet18(num_classes=10, img_dim=(32, 32))
(X_train, y_train), (X_test, y_test) = load_cifar()
X = np.concatenate((X_train, X_test), axis = 0)
N = X.shape[0]
N_train = X_train.shape[0]
N_test = X_test.shape[0]

predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, X_train,
                                                      y_train)
y_test_ntk = predict_fn(x_test=X_test, get='ntk')

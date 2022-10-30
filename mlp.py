from jax import value_and_grad, numpy as jnp, random, jit
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

key = random.PRNGKey(2022)  # random key

k1, k2 = random.split(key, num=2)

# generate dataset
X, y = datasets.make_circles(n_samples=1000, noise=0.2, factor=0.5)

#standardize
X -= X.mean()
X /= X.std()

plt.figure()
plt.scatter(x=X[:, 0], y=X[:, 1], c=y)
plt.show()

y = np.expand_dims(y, axis=1)
# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)


class Linear:
    def __init__(self, feature_in, feature_out):
        self.feature_in = feature_in
        self.feature_out = feature_out

    def compute(self, x, W, b):
        return x @ W + b


@jit
def relu(x):
    return jnp.maximum(0, x)


@jit
def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


@jit
def bce_loss(hat_y, y):
    return -(y * jnp.log(hat_y) + (1 - y) * jnp.log(1 - hat_y))


# hyperparameters
lr = 0.1
batch_size = 1  # 1 for SGD, X for mini-batch GD
model = [
    Linear(feature_in=2, feature_out=16),
    relu,
    Linear(feature_in=16, feature_out=1),
    sigmoid,
]


class UnknownLayer(Exception):
    pass


# build params
params = {}
for idx, layer in enumerate(model):
    _, k1 = random.split(k1)
    if isinstance(layer, Linear):
        feature_in = layer.feature_in
        feature_out = layer.feature_out
        params[f"linear_{idx}"] = random.normal(k1, (feature_in, feature_out)) * 0.01
        params[f"bias_{idx}"] = jnp.zeros(feature_out)
    elif layer in [relu, sigmoid]:
        pass  # no params needed
    else:
        raise UnknownLayer(f"{layer} is unknown")


@jit
def forward(params, X):
    out = X
    for idx, layer in enumerate(model):
        if isinstance(layer, Linear):
            out = layer.compute(out, params[f"linear_{idx}"], params[f"bias_{idx}"])
        elif layer in [relu, sigmoid]:
            out = layer(out)
    return out


@jit
def GD(params, params_grad):
    for k in params.keys():
        params[k] -= lr * params_grad[k]
    return params


@jit
def loss(params, X, y):
    y_hat = forward(params, X)
    loss_value = bce_loss(y_hat, y)
    return loss_value.mean()


enhanced_loss = value_and_grad(loss)

# train
for epoch in range(3001):
    _, k2 = random.split(k2)
    idx = random.randint(k2, (batch_size,), 0, X_train.shape[0])
    loss_value, params_grad = enhanced_loss(
        params, X_train[idx], y_train[idx]
    )  # forward

    if epoch % 500 == 0:
        print(loss_value)

    params = GD(params, params_grad)  # gradient descent

# evaluate
preds = forward(params, X_test)
preds = preds.at[preds >= 0.5].set(1)
preds = preds.at[preds < 0.5].set(0)
acc = (preds == y_test).mean()
print("Accuracy :", acc)

# contourf
x = np.linspace(-2, 2, 1000)
y = np.linspace(-2, 2, 1000)
xx, yy = np.meshgrid(x, y)

grid = np.c_[xx.ravel(), yy.ravel()]

preds = forward(params, grid)

h = plt.contourf(x, y, preds.reshape(x.shape[0], -1))
plt.axis("scaled")
plt.colorbar()
plt.show()

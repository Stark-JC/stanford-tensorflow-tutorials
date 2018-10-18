import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()
x = tfe.Variable(2.0)  # 注意不是  tf.Variable


def loss(y):
    return (y - x ** 2) ** 2


grad = tfe.implicit_gradients(loss)

print(loss(7.))
# tf.Tensor(9.0, shape=(), dtype=float32)

print(grad(7.))
# [(<tf.Tensor: id=50, shape=(), dtype=float32, numpy=-24.0>,
#   <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>)]

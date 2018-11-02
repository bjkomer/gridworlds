import numpy as np
import nengo


def power(s, e):
    x = np.fft.ifft(np.fft.fft(s.v) ** e).real
    return nengo.spa.SemanticPointer(data=x)


def encode_point(x, y, x_axis_vec, y_axis_vec):

    return power(x_axis_vec, x) * power(y_axis_vec, y)


def unitary_vector(dim):
    vec = nengo.spa.SemanticPointer(dim)
    vec.make_unitary()
    # return vec.v
    return vec

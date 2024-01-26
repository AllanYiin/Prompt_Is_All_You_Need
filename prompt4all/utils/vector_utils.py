import numpy as np

__all__ = ['epsilon', 'ndim', 'numel', 'matmul', 'l2_normalize', 'element_cosine_distance']


def epsilon():
    """Method for access epsilon attribute in session

    Returns: a float

    Example
        >>> print(epsilon())
        1e-08

    """
    return 1e-7


def ndim(x):
    """Number of dimension of a tensor

    Args:
        x (np.ndarray): input tensor.

    Returns:
        (int) Number of dimension

    """
    return len(x.shape)


def numel(x: np.ndarray):
    """Number of elements of a tensor

    Args:
        x (Tensor): input tensor.

    Returns:
        (int) Number of elements

    """
    return x.size


def matmul(a, b, transpose_a=False, transpose_b=False):
    """Multiplies matrix `a` by matrix `b`, producing `a` * `b`.

     The inputs must, following any transpositions, be tensors of rank >= 2
     where the inner 2 dimensions specify valid matrix multiplication dimensions,
     and any further outer dimensions specify matching batch size.

     Both matrices must be of the same type. The supported types are:
     `float16`, `float32`, `float64`, `int32`, `complex64`, `complex128`.

     Either matrix can be transposed or adjointed (conjugated and transposed) on
     the fly by setting one of the corresponding flag to `True`. These are `False`
     by default.

     If one or both of the matrices contain a lot of zeros, a more efficient
     multiplication algorithm can be used by setting the corresponding
     `a_is_sparse` or `b_is_sparse` flag to `True`. These are `False` by default.
     This optimization is only available for plain matrices (rank-2 tensors) with
     datatypes `bfloat16` or `float32`.

     A simple 2-D tensor matrix multiplication:


     >>> a =reshape(np.array([1, 2, 3, 4, 5, 6]),[2, 3])
     >>> a  # 2-D tensor
     tensor([[1, 2, 3],
            [4, 5, 6]])
     >>> b = reshape(np.array([7, 8, 9, 10, 11, 12]), [3, 2])
     >>> b  # 2-D tensor
     tensor([[ 7,  8],
            [ 9, 10],
            [11, 12]])
     >>> c = matmul(a, b)
     >>> c  # `a` * `b`
     tensor([[ 58,  64],
            [139, 154]])

     A batch matrix multiplication with batch shape [2]:

     >>> a =  reshape(np.array(np.arange(1, 13, dtype=np.int32)),[2, 2, 3])
     >>> a  # 3-D tensor
     tensor([[[ 1,  2,  3],
             [ 4,  5,  6]],
            [[ 7,  8,  9],
             [10, 11, 12]]])
     >>> b =  reshape(np.array(np.arange(13, 25, dtype=np.int32)),[2, 3, 2])
     >>> b  # 3-D tensor
     tensor([[[13, 14],
             [15, 16],
             [17, 18]],
            [[19, 20],
             [21, 22],
             [23, 24]]])
     >>> c = matmul(a, b)
     >>> c  # `a` * `b`
     tensor([[[ 94, 100],
             [229, 244]],
            [[508, 532],
             [697, 730]]])

     Since python >= 3.5 the @ operator is supported
     (see [PEP 465](https://www.python.org/dev/peps/pep-0465/)). In TensorFlow,
     it simply calls the `np.matmul()` function, so the following lines are
     equivalent:

        >>> d = a @ b @ [[10], [11]]
        >>> d = matmul(np.matmul(a, b), [[10], [11]])

     Args:
       a: `Tensor` and rank > 1.
       b: `Tensor` with same type and rank as `a`.
       transpose_a: If `True`, `a` is transposed before multiplication.
       transpose_b: If `True`, `b` is transposed before multiplication.


     Returns:
       A `Tensor` of the same type as `a` and `b` where each inner-most matrix
       is the product of the corresponding matrices in `a` and `b`, e.g. if all
       transpose or adjoint attributes are `False`:

       `output[..., i, j] = sum_k (a[..., i, k] * b[..., k, j])`,
       for all indices `i`, `j`.

       Note: This is matrix product, not element-wise product.


     Raises:
       ValueError: If `transpose_a` and `adjoint_a`, or `transpose_b` and
         `adjoint_b` are both set to `True`.

     """
    if transpose_a:
        a = a.T
    if transpose_b:
        b = b.T
    return np.matmul(a, b)


def l2_normalize(x: np.ndarray, axis=1, keepdims=True, eps=epsilon()):
    """

    Args:
        x (np.ndarray): input tensor.

    Returns:
        (np.ndarray): output tensor and have same shape with x.



    """
    if ndim(x) == 1:
        axis = 0
    return x / np.sqrt(np.sum(np.square(x), axis=axis, keepdims=keepdims) + eps)


def element_cosine_distance(v1, v2, axis=-1):
    """

    Args:
        v1 ():
        v2 ():
        axis ():

    Returns:

    """
    x_normalized = l2_normalize(v1, axis=axis, keepdims=True)
    y_normalized = l2_normalize(v2, axis=axis, keepdims=True)

    cos = matmul(x_normalized, y_normalized, False, True)

    return cos

import torch
from typing import NamedTuple

def tonp(x):
    return x.detach().cpu().numpy()

def frnp(x, device=None):
    return torch.Tensor(x).to(device)

# Sorting and parsing
class TopKReturnType(NamedTuple):
    values: torch.Tensor
    indices: torch.Tensor

def topk(vec: torch.Tensor, k: int):
    assert vec.ndim == 1, "'vec' must be a 1D array"
    assert k > 0, "k should be greater than 0"
    sort_perm = torch.flip(torch.argsort(vec), dims=[0])  # descending order
    indices = sort_perm[:k]
    values = vec[indices]
    return TopKReturnType(values, indices)

def sqrtmh(A: torch.Tensor):
    # Credits to
    """Compute the square root of a Symmetric or Hermitian positive definite matrix or batch of matrices.
    Credits to  `https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228
    <https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228>`_."""
    L, Q = torch.linalg.eigh(A)
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH

def random_split(X, Y, n):
    """
    Randomly splits the data X into n partitions with equal size.

    Parameters:
        X (array-like): The input data.
        n (int): The number of random splits.

    Returns:
        list: List of partitions.
    """
    res = (X.shape[0] % n)
    if res != 0:
        X = X[:-res]
        Y = Y[:-res]
    batch_size = X.shape[0]
    idxs = torch.randperm(batch_size) # Randomly shuffle the indices
    X, Y = X[idxs], Y[idxs] # Shuffle the data

    batch_size = X.shape[0]
    split_size = batch_size // n # Size of each split

    splits_X = [X[idxs[i*split_size:(i+1)*split_size]] for i in range(n - 1)]  # Create n splits
    splits_X.append(X[idxs[(n-1)*split_size:]])  # Add the last split with the remaining elements

    splits_Y = [Y[idxs[i * split_size:(i + 1) * split_size]] for i in range(n - 1)]  # Create n splits
    splits_Y.append(Y[idxs[(n - 1) * split_size:]])  # Add the last split with the remaining elements

    return tuple(splits_X) + tuple(splits_Y)

def cross_cov(A, B, rowvar=True, bias=False, centered=True):
    """Cross covariance of two matrices.

    Args:
        A (np.ndarray or torch.Tensor): Matrix of size (n, p).
        B (np.ndarray or torch.Tensor): Matrix of size (n, q).
        rowvar (bool, optional): Whether to calculate the covariance along the rows. Defaults to False.

    Returns:
        np.ndarray or torch.Tensor: Matrix of size (p, q) containing the cross covariance of A and B.
    """
    if rowvar is False:
        A = A.T
        B = B.T

    if centered:
        A = A - A.mean(axis=1, keepdims=True)
        B = B - B.mean(axis=1, keepdims=True)

    C = A @ B.T

    if bias:
        return C / A.shape[1]
    else:
        return C / (A.shape[1] - 1)

def filter_reduced_rank_svals(values, vectors):
    eps = 2 * torch.finfo(torch.get_default_dtype()).eps
    # Filtering procedure.
    # Create a mask which is True when the real part of the eigenvalue is negative or the imaginary part is nonzero
    is_invalid = torch.logical_or(torch.real(values) <= eps,
                                  torch.imag(values) != 0
                                  if torch.is_complex(values)
                                  else torch.zeros(len(values), device=values.device))
    # Check if any is invalid take the first occurrence of a True value in the mask and filter everything after that
    if torch.any(is_invalid):
        values = values[~is_invalid].real
        vectors = vectors[:, ~is_invalid]

    sort_perm = topk(values, len(values)).indices
    values = values[sort_perm]
    vectors = vectors[:, sort_perm]

    # Assert that the eigenvectors do not have any imaginary part
    assert torch.all(
        torch.imag(vectors) == 0 if torch.is_complex(values) else torch.ones(len(values))
    ), "The eigenvectors should be real. Decrease the rank or increase the regularization strength."

    # Take the real part of the eigenvectors
    vectors = torch.real(vectors)
    values = torch.real(values)
    return values, vectors

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches
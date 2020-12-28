import torch
import math

def weighted_sigmoid_cross_entropy_with_logits(
    labels,
    logits,
    positive_weights=1.0,
    negative_weights=1.0
):
    """Computes a weighting of sigmoid cross entropy given `logits`.

    Measures the weighted probability error in discrete classification tasks in
    which classes are independent and not mutually exclusive.  For instance, one
    could perform multilabel classification where a picture can contain both an
    elephant and a dog at the same time. The class weight multiplies the
    different types of errors.
    For brevity, let `x = logits`, `y = labels`, `c = positive_weights`,
    `d = negative_weights`  The
    weighed logistic loss is
    
    ```
    c * y * -log(sigmoid(x)) + d * (1 - y) * -log(1 - sigmoid(x))
    = c * y * log (1 + exp(-x)) + d * (1 - y) * -log (exp(-x) / ( 1 + exp(-x)))
    = c * y * log (1 + exp(-x)) + d * (1 - y) * log ( 1 + exp(-x))) + d * (1-y) * x
    = [c * y + d * (1-y)] * log (1 + exp(-x)) + d * (1-y) * x
    ```
    
    To ensure numerical stability and avoid overflow, we implement log (1 + exp(-x)) as
        log(1+exp(-x)) = max(0, -x) + log (1 + exp(-abs(x)))
    
    Note that the loss is NOT an upper bound on the 0-1 loss, unless it is divided
    by log(2). When x=0, -log(sigmoid(x)) = -log(1- sigmoid(x)) = log(2). While 0-1 loss
    passes point (0, 1).

    Parameters
    ----------
    labels : Tensor of float32.
        [description]
    logits : Tensor of float32.
        [description]
    positive_weights : float, optional
        [description], by default 1.0
    negative_weights : float, optional
        [description], by default 1.0
    """
    softplus_term = torch.log(1.0 + torch.exp(-torch.abs(logits))) + torch.maximum(-logits, torch.zeros_like(logits))
    
    weight_left = positive_weights * labels + negative_weights * (1.0 - labels)
    loss = weight_left * softplus_term + negative_weights * logits * (1.0 - labels)
    return loss / math.log(2)


def weighted_hinge_loss(
    labels,
    logits,
    positive_weights=1.0,
    negative_weights=1.0
):
    """Computes weighted hinge loss given logits `logits`.

    The loss applies to multi-label classification tasks where labels are
    independent and not mutually exclusive. See also
    `weighted_sigmoid_cross_entropy_with_logits`.

    Parameters
    ----------
    labels : `Tensor` of `float32`.
        Each entry must be either 0 or 1.
    logits : [type]
        [description]
    positive_weights : float, optional
        [description], by default 1.0
    negative_weights : float, optional
        [description], by default 1.0
    """
    
    positives_term = positive_weights * labels * torch.maximum(1.0 - logits, torch.zeros_like(logits))
    negatives_term = negative_weights * (1.0 - labels) * torch.maximum(1.0 + logits, torch.zeros_like(logits))
    return positives_term + negatives_term


def weighted_surrogate_loss(
    labels,
    logits,
    surrogate_type='xent',
    positive_weights=1.0,
    negative_weights=1.0
):
    """A wrapper of the above two surrogate loss.

    Parameters
    ----------
    labels : [type]
        [description]
    logits : [type]
        [description]
    surrogate_type : str, optional
        [description], by default 'xent'
    positive_weights : float, optional
        [description], by default 1.0
    negative_weights : float, optional
        [description], by default 1.0

    Returns
    -------
    loss :
        Surrogate loss.

    Raises
    ------
    ValueError
        surrogate type not supported.
    """
    if surrogate_type == 'xent':
        surrogate = weighted_sigmoid_cross_entropy_with_logits
    elif surrogate_type == 'hinge':
        surrogate = weighted_hinge_loss
    else:
        raise ValueError('surrogate_type {} not supported.'.format(surrogate_type))

    return surrogate(
            logits=logits,
            labels=labels,
            positive_weights=positive_weights,
            negative_weights=negative_weights
        )


def build_label_priors(
    labels,
    weights=None,
    positive_pseudocount=1.0,
    negative_pseudocount=1.0
):
    """Track the running label prior probabilities, (i.e. the fraction
    of the positive sample in training data).
    
    For each label i, the label priors are estimated as
        (P + \sum_i w_i y_i) / (P + N + \sum w_i),
    where for a sample in the mini-batch, y_i is the ith label, w_i is the ith weight,
    P is a pseudo-count of positive labels, and N is a pseudo-count of negative labels.

    Parameters
    ----------
    labels : `Tensor` of `float32` with shape [batch_size, num_labels].
        Entries should be in [0, 1].
    weights : None or `Tensor` or `float32`, optional
        Coefficients representing the weight of each label. Must be either
        a `Tensor` of shape [batch_size, num_labels] or `None`, in which case each
        weight is by default 1.0.
    positive_pseudocount : float, optional
        Number of positive labels to initialize label priors, by default 1.0
    negative_pseudocount : float, optional
        Number of negative labels to initialize label priors, by default 1.0

    Returns
    -------
    label_priors: 1d `Tensor` of shape [num_labels] representing the weighted
        label priors.
    """
    num_labels = get_num_labels(labels_or_logits=labels)
    
    if weights is None:
        weights = torch.ones_like(labels)
    
    # positive ones
    weighted_label_counts = torch.tensor([positive_pseudocount]*num_labels, dtype=torch.float32)
    weighted_label_counts_update = torch.sum(weights * labels, dim=0) + weighted_label_counts
    
    # all
    weighted_sum = torch.tensor([positive_pseudocount + negative_pseudocount]*num_labels, dtype=torch.float32)
    weighted_sum_update = torch.sum(weights, dim=0) + weighted_sum
    
    label_priors = weighted_label_counts_update / weighted_sum_update
    return label_priors


def get_num_labels(labels_or_logits):
    """Get the number of classes.

    Parameters
    ----------
    labels_or_logits : `Tensor` of `float32`.
        labels or logits in shape [batch_size, num_labels].

    Returns
    -------
    int
        number of classes.
    """
    if labels_or_logits.dim() <=1:
        return 1
    return labels_or_logits.size(1)


if __name__ == "__main__":
    labels = torch.tensor([[0, 0, 1],
                           [0, 1, 0],
                           [1, 0, 0]])
    logits = torch.randn(3, 3)
    priors = build_label_priors(labels)
    print(priors)
    
    l = weighted_sigmoid_cross_entropy_with_logits(labels, logits)
    print(l)
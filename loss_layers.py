import math
import torch
import torch.nn.functional as F
from util import weighted_surrogate_loss, build_label_priors

def _create_non_negative_lambda(size=1):
    x = torch.nn.Parameter(size)
    return F.softplus(x)
    

class GradientReversedLambda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lambda_input):
        # do nothing return the lambda_input itself (just like what nn.Idendity() does).
        return lambda_input
        # ctx.save_for_backward(lambda_input)
        # lambda_output = lambda_input
        # return lambda_output
    
    @staticmethod
    def backward(ctx, grad_output):
        # do nothing but reverse the gradient.
        return -grad_output


def recall_at_precision_loss(
    labels,
    logits,
    target_precision,
    dual_variable,
    weights=1.0,
    label_priors=None,
    surrogate_type='xent'
):
    """Compute the recall at a given precision.

    Parameters
    ----------
    labels : `Tensor` of float32.
        [description]
    logits : [type]
        [description]
    target_precision : [type]
        [description]
    dual_variable : [type]
        [description]
    weights : float, optional
        [description], by default 1.0
    label_priors : [type], optional
        [description], by default None
    surrogate_type : str, optional
        [description], by default 'xent'

    Returns
    -------
    loss
        [description]
    """
    labels, logits, weights, original_shape = _prepare_labels_logits_weights(labels, logits, weights)

    if label_priors is None:
        label_priors = build_label_priors(labels=labels, weights=weights)

    weighted_loss = weights * weighted_surrogate_loss(
        labels=labels,
        logits=logits,
        surrogate_type=surrogate_type,
        positive_weights=1.0 + dual_variable * (1.0 - target_precision),
        negative_weights=dual_variable * target_precision
    )
    # one dual varialbe for each label
    dual_variable_term = dual_variable * (1.0 - target_precision) * label_priors
    # note that weighted_loss is of shape [batch_size, num_labels],
    # dual_variable_term is of shape [num_labels], then subtraction will incur
    # broadcasting where shape [num_labels] changed to [-1, num_labels]. 
    loss = weighted_loss - dual_variable_term
    return loss.view(*original_shape)


def precision_at_recall_loss(
    labels,
    logits,
    target_recall,
    dual_variable,
    weights=1.0,
    label_priors=None,
    surrogate_type='xent'
):
    labels, logits, weights, original_shape = _prepare_labels_logits_weights(labels, logits, weights)

    if label_priors is None:
        label_priors = build_label_priors(labels=labels, weights=weights)

    weighted_loss = weights * weighted_surrogate_loss(
        labels=labels,
        logits=logits,
        surrogate_type=surrogate_type,
        positive_weights=dual_variable,
        negative_weights=1.0
    )
    
    lambda_term = dual_variable * label_priors * (target_recall - 1.0)
    
    loss = weighted_loss + lambda_term
    return loss.view(*original_shape)


def _prepare_labels_logits_weights(
    labels,
    logits,
    weights
):
    assert labels.size() == logits.size(), 'logits and labels should have the same size.'
    original_shape = labels.size()
    # if labels.dim() > 0:
    #     original_shape[0] = -1

    if labels.dim() <= 1:
        labels = labels.view(-1, 1)
        logits = logits.view(-1, 1)

    if isinstance(weights, torch.Tensor) and weights.dim() == 1:
        # weights have shape [batch_size]. Reshape to [batch_size, 1]
        weights = weights.view(-1, 1)

    if isinstance(weights, float):
        # weights is scalar. Construct to match logits.
        weights *= torch.ones_like(logits)
    return labels, logits, weights, original_shape


if __name__ == "__main__":
    x = torch.tensor([1, 2, 0, -2, -1.])
    var_x = torch.nn.Parameter(x)
    non_negative_x = F.softplus(var_x)

    non_negative_x.retain_grad()
    if False:
        rev_f = GradientReversedLambda.apply        
        non_negative_x_ = rev_f(non_negative_x)
    else:
        non_negative_x_ = non_negative_x

    s = non_negative_x_.sum()
    s.backward(retain_graph=True)
    
    print('non_negative_x grad', non_negative_x.grad)
    print('var_x grad', var_x.grad)
    
    
    
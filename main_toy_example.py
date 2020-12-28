import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import precision_score, roc_auc_score, recall_score

import loss_layers

TARGET_RECALL=0.96

DATA_CONFIG = {
        'positives_centers': [[0., 1.0], [1, -0.5]],
        'negatives_centers': [[0., -0.5], [1, 1.0]],
        'positives_variances': [0.15, 0.1],
        'negatives_variances': [0.15, 0.1],
        'positives_counts': [500, 50],
        'negatives_counts': [3000, 100],
}


class Model(torch.nn.Module):
    def __init__(self, d_in=2, d_out=1) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(d_in, d_out)
        
    def forward(self, x):
        return self.lin(x)


def create_training_and_eval_data_for_experiment(**data_config):
    """Creates train and eval data sets.
    Note: The synthesized binary-labeled data is a mixture of four Gaussians - two
        positives and two negatives. The centers, variances, and sizes for each of
        the two positives and negatives mixtures are passed in the respective keys
        of data_config:
    Args:
        **data_config: Dictionary with Array entries as follows:
            positives_centers - float [2,2] two centers of positives data sets.
            negatives_centers - float [2,2] two centers of negatives data sets.
            positives_variances - float [2] Variances for the positives sets.
            negatives_variances - float [2] Variances for the negatives sets.
            positives_counts - int [2] Counts for each of the two positives sets.
            negatives_counts - int [2] Counts for each of the two negatives sets.
    Returns:
        A dictionary with two shuffled data sets created - one for training and one
        for eval. The dictionary keys are 'train_data', 'train_labels', 'eval_data',
        and 'eval_labels'. The data points are two-dimentional floats, and the
        labels are in {0,1}.
    """
    def data_points(is_positives, index):
        variance = data_config['positives_variances'
                           if is_positives else 'negatives_variances'][index]
        center = data_config['positives_centers'
                         if is_positives else 'negatives_centers'][index]
        count = data_config['positives_counts'
                                                if is_positives else 'negatives_counts'][index]
        return variance*np.random.randn(count, 2) + np.array([center])

    def create_data():
        return np.concatenate([data_points(False, 0),
                           data_points(True, 0),
                           data_points(True, 1),
                           data_points(False, 1)], axis=0)

    def create_labels():
        """Creates an array of 0.0 or 1.0 labels for the data_config batches."""
        return np.array([0.0]*data_config['negatives_counts'][0] +
                                        [1.0]*data_config['positives_counts'][0] +
                                        [1.0]*data_config['positives_counts'][1] +
                                        [0.0]*data_config['negatives_counts'][1])

    permutation = np.random.permutation(
            sum(data_config['positives_counts'] + data_config['negatives_counts']))

    train_data = create_data()[permutation, :]
    eval_data = create_data()[permutation, :]
    train_labels = create_labels()[permutation]
    eval_labels = create_labels()[permutation]

    return {
            'train_data': train_data,
            'train_labels': train_labels,
            'eval_data': eval_data,
            'eval_labels': eval_labels
    }


class NonNegativeLambda(torch.nn.Module):
    def __init__(self, num_labels) -> None:
        super().__init__()
        self.x = torch.nn.Parameter(torch.tensor([0.] * num_labels))

    def forward(self):
        out = F.softplus(self.x)
        return loss_layers.GradientReversedLambda.apply(out)


def run(model, x, y, use_global_objectives=True):
    x = torch.tensor(x).float()
    y = torch.tensor(y).float()
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset=dataset, batch_size=200, shuffle=True, num_workers=4)

    with torch.autograd.set_detect_anomaly(True):
        if use_global_objectives:
            num_labels = loss_layers.get_num_labels(y)
            dual_variable_model = NonNegativeLambda(num_labels)
            optimizer = torch.optim.SGD(list(model.parameters()) + list(dual_variable_model.parameters()), lr=0.1, momentum=0.9)

            # optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
            
            for epoch in range(100):
                for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
                    logits = model(batch_x).squeeze(dim=-1)
                    dual_variable = dual_variable_model()
                    loss = loss_layers.precision_at_recall_loss(batch_y, logits, TARGET_RECALL, dual_variable).mean()
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
            for epoch in range(100):
                for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
                    logits = model(batch_x).squeeze(dim=-1)
                    loss = F.binary_cross_entropy_with_logits(logits, batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
    return model


        
def train_model(data, use_global_objectives=True):
    
    def precision_at_recall(scores, labels, target_recall):
        positive_scores = scores[labels == 1.0]
        threshold = np.percentile(positive_scores, 100 - target_recall * 100)
        predicted = scores >= threshold
        return precision_score(labels, predicted)
    
    linear = Model()
    linear = run(linear, data['train_data'], data['train_labels'], use_global_objectives=use_global_objectives)
    scores = linear(torch.tensor(data['eval_data']).float()).squeeze(dim=-1).detach().numpy()
    
    precision = precision_at_recall(scores, labels=data['eval_labels'], target_recall=TARGET_RECALL)
    roc_score = roc_auc_score(data['eval_labels'], scores)
    acc = np.mean((scores > 0.5) == data['eval_labels'])
    return precision, roc_score, acc
    
    
if __name__ == "__main__":
    experiment_data = create_training_and_eval_data_for_experiment(
      **DATA_CONFIG)
    precision, roc, acc = train_model(experiment_data, use_global_objectives=True)
    print('==> global loss precision: {:.4f}, roc: {:.4f}, acc: {:.4f}'.format(precision, roc, acc))
    
    precision, roc, acc = train_model(experiment_data, use_global_objectives=False)
    print('==> normal precision: {:.4f}, roc: {:.4f}, acc: {:.4f}'.format(precision, roc, acc))
    
    
    

    
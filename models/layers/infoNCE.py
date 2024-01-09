import torch
import torch.nn.functional as F

batch_size=24
n_views=5
temperature=1

def info_nce_loss(features,lables,device):
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    # print(labels)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to('cpu')
    print(labels.shape,labels)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    print(labels.shape,labels)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    print(positives.shape,negatives.shape)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to('cpu')
    print(logits.shape,logits,labels.shape,labels)

    logits = logits / temperature
    return logits, labels

if __name__=='__main__':
    features = torch.randn(24,5,512)
    info_nce_loss(features,None,'cpu')
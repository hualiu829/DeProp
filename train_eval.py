import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator


def train(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(x=data.x, adj_t=data.adj_t)[train_idx]
    if len(data.y.shape) == 1:
        y = data.y[train_idx]
    else:
        y = data.y.squeeze(1)[train_idx]  ## for ogb data
    loss = F.nll_loss(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()



@torch.no_grad()
def test(model, data, split_idx, args):
    model.eval()
    with torch.no_grad():
        if args.smooth:
            out, sim, corr=model(x=data.x, adj_t=data.adj_t, test_true=True)
        else:
            out = model(x=data.x, adj_t=data.adj_t)


    y_pred = out.argmax(dim=-1, keepdim=True)
    if len(data.y.shape) == 1:
        y = data.y.unsqueeze(dim=1) # for non ogb datas
    else:
        y = data.y

    evaluator = Evaluator(name='ogbn-arxiv')
    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    if args.smooth:
        return (train_acc, valid_acc, test_acc), sim, corr
    else:
        return (train_acc, valid_acc, test_acc)




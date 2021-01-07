import torch
import random
import numpy as np
from torch.autograd import grad


def hessian_vector_product(ys, xs, v):
    J = grad(ys, xs, create_graph=True)[0]
    grads = grad(J, xs, v, retain_graph=True)
    del J, ys, v
    torch.cuda.empty_cache()
    return grads


def lissa(train_loss, test_loss, layer_weight, model):
    scale = 10
    damping = 0.1
    num_samples = 1
    v = grad(test_loss, layer_weight)[0]
    cur_estimate = v.clone()
    prev_norm = 1
    diff = prev_norm
    count = 0
    while diff > 0.00001 and count < 10000:
        try:
            hvp = hessian_vector_product(train_loss, layer_weight, cur_estimate)
            cur_estimate = [a + (1 - damping) * b - c / scale for (a, b, c) in zip(v, cur_estimate, hvp)]
            cur_estimate = torch.squeeze(torch.stack(cur_estimate)).view(1, -1)
            model.zero_grad()
            numpy_est = cur_estimate.detach().cpu().numpy()
            numpy_est = numpy_est.reshape(1, -1)

            if (count % 100 == 0):
                print("Recursion at depth %s: norm is %.8lf" % (count, np.linalg.norm(np.concatenate(numpy_est))))
            count += 1
            diff = abs(np.linalg.norm(np.concatenate(numpy_est)) - prev_norm)
            prev_norm = np.linalg.norm(np.concatenate(numpy_est))
            ihvp = [b / scale for b in cur_estimate]
            ihvp = torch.squeeze(torch.stack(ihvp))
            ihvp = [a / num_samples for a in ihvp]
            ihvp = torch.squeeze(torch.stack(ihvp))
        except Exception:
            print('LiSSA Failed')
            return np.zeros_like(v.detach().cpu().numpy())
    return ihvp.detach()


def VDP_influence(x_train, y_train, x_test, y_test, model, layer_weight):
    eqn_5 = []

    model.eval()
    mu, sigma = model(x_train.to('cuda:0'))
    train_loss = model.batch_loss(mu, sigma, y_train.view(-1, 1).to('cuda:0'))
    mu, sigma = model(torch.tensor(x_test).float().to('cuda:0'))
    test_loss = model.batch_loss(mu, sigma, torch.from_numpy(y_test).view(-1, 1).float().to('cuda:0'))

    ihvp = lissa(train_loss, test_loss, layer_weight, model)

    x = x_train
    x.requires_grad = True
    mu, sigma = model(x.to('cuda:0'))
    x_loss = model.batch_loss(mu, sigma, y_train.view(-1, 1).to('cuda:0'))
    grads = grad(x_loss, layer_weight, create_graph=True)[0]
    grads = grads.squeeze()
    grads = grads.view(1, -1).squeeze()
    infl = (torch.dot(ihvp.view(-1, 1).squeeze(), grads)) / len(x_train)
    i_pert = grad(infl, x, retain_graph=False)
    i_pert = i_pert[0]

    # eqn_2 = -infl.detach().cpu().numpy()
    eqn_5.append(np.sum(-i_pert.detach().cpu().numpy(), axis=0))
    model.zero_grad()

    return np.asarray(eqn_5)

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
    damping = 0.01
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


def influence(x_train, y_train, x_test, y_test, model, layer_weight, n=1, std=0.2,
              criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([14.80], device='cuda:0')), device='cuda:0'):
    eqn_5 = []
    x_train, y_train = torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()
    x_test, y_test = torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float()
    for itr in range(n):
        if n > 1:
            np.random.seed(random.randint(0, 10000000))
            noise = np.random.normal(0, std, x_test.size())
            x_test = x_test.cpu() + noise  # add noise to test data
        if device == 'cuda:0':
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            model = model.to(device)
        train_loss = criterion(model(x_train), y_train.view(-1, 1))
        test_loss = criterion(model(x_test), y_test.view(-1, 1))

        ihvp = lissa(train_loss, test_loss, layer_weight, model)

        x = x_train
        x.requires_grad = True
        x_out = model(x)
        x_loss = criterion(x_out, y_train.view(-1, 1))
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

import copy
import torch

def FedAvg(w, client_freq):
    w_avg = copy.deepcopy(w[0])
    # for k in w_avg.keys():
    #     for i in range(1, len(w)):
    #         #print('done')
    #         w_avg[k] += w[i][k]*client_freq[i]
    #     w_avg[k] = torch.div(w_avg[k], len(w))
    num_takepart = len(w)
    ratio_takepart = sum(client_freq[:num_takepart])
    ratio = [freq / ratio_takepart for freq in client_freq[:num_takepart]]
    for net_id in range(num_takepart):
        if net_id == 0:
            for key in w[net_id]:
                w_avg[key] = w[net_id][key] * ratio[net_id]
        else:
            for key in w[net_id]:
                w_avg[key] += w[net_id][key] * ratio[net_id]
    return w_avg


def model_dist(w_1, w_2):
    assert w_1.keys() == w_2.keys(), "Error: cannot compute distance between dict with different keys"
    dist_total = torch.zeros(1).float()
    for key in w_1:
        dist = torch.norm(w_1[key].cpu() - w_2[key].cpu())
        dist_total += dist.cpu()

    return dist_total.cpu().item()


def update_global_ema(w, ema_w, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    assert w.keys() == ema_w.keys(), 'Error: aggregating models with different keys'
    # w_after = copy.deepcopy(w)
    for key in w:
        w[key]=w[key].cpu()
        ema_w[key].mul_(alpha).add_(w[key], alpha=1 - alpha)
import numpy as np

seqToArray = lambda x: np.reshape(x, (x.shape[0] * x.shape[1], -1))
def arrayToSeq(x, numEp, epLen):
    if type(x) is not np.ndarray:
        x = x.cpu().detach().numpy()
    return np.reshape(x, (numEp, epLen, -1))

def norm(x,data,tar_type='targets'):
    if type(x) is not np.ndarray:
        x = x.cpu().detach().numpy()
    if tar_type=='observations':
        return data.normalize(x, data.normalization["observations"][0],
                   data.normalization["observations"][1])
    if tar_type == 'actions':
        return data.normalize(x, data.normalization["actions"][0],
                              data.normalization["actions"][1])

    else:
        return data.normalize(x, data.normalization["targets"][0],
                       data.normalization["targets"][1])



def denorm(x, data, tar_type='targets'):
    if type(x) is not np.ndarray:
        x = x.cpu().detach().numpy()
    if tar_type=='observations':
        return data.denormalize(x, data.normalization["observations"][0],
                   data.normalization["observations"][1])
    if tar_type == 'actions':
        return data.denormalize(x, data.normalization["actions"][0],
                                data.normalization["actions"][1])
    if tar_type == 'act_diff':
        return data.denormalize(x, data.normalization["act_diff"][0],
                                data.normalization["act_diff"][1])


    else:
        return data.denormalize(x, data.normalization["diff"][0],
                       data.normalization["diff"][1])


def diffToState(diff,current,data,standardize=True):
    '''
    :param diff: difference between next and current state
    :param current: current state
    :param data: data object
    :return: normalized next state
    '''
    if type(diff) is not np.ndarray:
        diff = diff.cpu().detach().numpy()
    if type(current) is not np.ndarray:
        current = current.cpu().detach().numpy()

    if standardize:
        current = denorm(current, data, 'observations')
        diff = denorm(diff, data, "diff")

        next = norm(current + diff, data, "observations")
    else:
        next = current + diff

    return next,diff

def diffToAct(diff,prev,data,standardize=True):
    '''
    :param diff: difference between next and current state
    :param current: current state
    :param data: data object
    :param standardize: if the data is standardized apriori or not
    :return: normalized next state
    '''
    if type(diff) is not np.ndarray:
        diff = diff.cpu().detach().numpy()
    if type(prev) is not np.ndarray:
        prev = prev.cpu().detach().numpy()

    if standardize:
        prev = denorm(prev, data, 'actions')
        diff = denorm(diff, data, "act_diff")

        current = norm(prev + diff, data, "actions")
    else:
        current = prev + diff

    return current,diff




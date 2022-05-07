import numpy as np

def interpolate(x1, y1, x2, y2, x):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m * x + b

def precision_at(recall, results, relevant):
    k = recall*len(relevant)
    x1, x2 = int(k)/len(relevant), (int(k) + 1)/len(relevant)
    y1, y2, cnt, index = 0, 0, 0, 0

    while index < len(results) and cnt < x1*len(relevant):
        if results[index] in relevant:
            cnt += 1
        index += 1
    if index == 0:
        y1 = 1
    else:
        y1 = cnt/index
    if int(k) == k:
        return y1

    while index < len(results) and cnt < x2*len(relevant):
        if results[index] in relevant:
            cnt += 1
        index += 1
    y2 = cnt/index
    return interpolate(x1, y1, x2, y2, recall)


# result: ranked song list returned by recommendation system
# relevant: songs in test users hidden song list 

## Accuracy score:
def acc_score(result, relevant):
    return len(list(set(result)&set(relevant)))/len(result)

## Mean precision at 0.25, 0.5, 0.75 recall levels:
def mean_precision(results, relevant):
    return (precision_at(0.25, results, relevant) +
        precision_at(0.5, results, relevant) +
        precision_at(0.75, results, relevant)) / 3

## Normalized precision:
def norm_precision(results, relevant):
    n, rel = len(results), len(relevant)
    rel_rank = sum([np.log(results.index(i)+1) for i in relevant])
    pos = sum([np.log(i+1) for i in range(rel)])
    dev = rel_rank - pos
    deno = n*np.log(n)-(n-rel)*np.log(n-rel)-rel*np.log(rel)
    return 1 - dev/deno
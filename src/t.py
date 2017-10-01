import math
import numpy as np

if __name__ == '__main__':
    with open('./data/stage1_solution_filtered.csv', 'rb') as f:
        stage1 = {}
        for line in f.readlines()[1:]:
            d = line.strip().split(',')
            id = d[0]
            stage1[id] = [float(x) for x in d[1:]]

    with open('./submissions/old/hatt.txt', 'rb') as f:
        submission = {}
        for line in f.readlines()[1:]:
            d = line.strip().split(',')
            id = d[0]
            submission[id] = [float(x) + 0.02 for x in d[1:]]
            # prediction = np.argmax(submission[id])
            # submission[id] = [1.0/9] * 9
            # submission[id] = [0.7] * 9
            # if submission[id][prediction] > 0.9:
            #     if prediction == 0:
            #         p = [0.0] * 9
            #         p[prediction] = 1.0
            #         submission[id] = p

    log_sum = 0.0
    count = float(len(stage1))
    for id in stage1.keys():
        real = stage1[id]
        prediction = submission[id]
        sum = np.sum(prediction)
        prediction = [p/sum for p in prediction]
        for i in range(9):
            p = prediction[i]
            p = max(min(p, 1.0 - 10e-15), 10e-15)
            log_sum += real[i] * math.log(p)
    loss = - log_sum / count
    print loss
# 1.75744098021
# 1.53447464244
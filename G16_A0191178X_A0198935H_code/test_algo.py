import random
import time
from tqdm import tqdm

def test_algo(algo, events, art_feats, size_rate=None, learn_rate=0.9):

    start = time.time()
    G_learn = 0
    G_deploy = 0
    N_learn = 0
    N_deploy = 1

    exp_learns = []
    exp_deploy = []

    if size_rate is None:
        events = events
    else:
        events = random.sample(events, int(len(events)*size_rate/100))

    for i, event in enumerate(tqdm(events)):
    # for i, event in enumerate(events):
        dis = event[0]
        reward = int(event[1])
        user_feat = event[2]
        pool_idx = event[3]

        chosen_art = algo.choose_arm(N_learn+N_deploy, user_feat, pool_idx, art_feats)
        if chosen_art == dis:
            # update only when chosen is displayed
            if random.random() < learn_rate:
                # update with learn rate
                G_learn += reward
                N_learn += 1
                algo.update(dis, reward, user_feat, pool_idx, art_feats)
                exp_learns.append(G_learn/N_learn)

            else:
                # dont update
                G_deploy += reward
                N_deploy += 1
                exp_deploy.append(G_deploy/N_deploy)

    end = time.time()

    exc_time = round(end-start, 1)
    print(algo.algo, round(G_deploy/N_deploy, 4), exc_time)
    
    return exp_learns, exp_deploy
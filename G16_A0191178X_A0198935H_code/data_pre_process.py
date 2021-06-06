import numpy as np
import fileinput
import pickle
from tqdm import tqdm

def generate_data(filenames):

    art_ids = []
    art_feats = []
    events = []

    with fileinput.input(files=filenames) as f:
        for line in tqdm(f):
            if (len(line.split())-10) % 7 != 0:
                # some error data to ignore
                None
            else:
                cols = line.strip().split('|')

                # user data
                user_feat = [float(x[2:]) for x in cols[1].strip().split()[1:]]
                user_click = cols[0].strip().split()[2]

                # article data
                pool_idx = []
                pool_ids = []
                for i in range(2, len(cols)):
                    art_line = cols[i].strip().split()
                    art_id = int(art_line[0])
                    art_feat = [float(x[2:]) for x in art_line[1:]]
                    
                    if art_id not in art_ids:
                        art_ids.append(art_id)
                        art_feats.append(art_feat)

                    pool_idx.append(art_ids.index(art_id))
                    pool_ids.append(art_id)

                # event data
                events.append(
                        [
                            pool_ids.index(int(cols[0].strip().split()[1])),
                            user_click,
                            user_feat,
                            pool_idx
                        ]
                    )


    with open('data/art_feats.pkl', 'wb') as f:
        pickle.dump(art_feats, f)

    with open('data/events.pkl', 'wb') as f:
        pickle.dump(events, f)

if __name__ == '__main__':
    generate_data(('data/ydata-fp-td-clicks-v1_0.20090509',
                    'data/ydata-fp-td-clicks-v1_0.20090510'))


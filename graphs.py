import os
import ipdb
import pickle
import matplotlib.pyplot as plt
import numpy as np


graphs = [
        ('test_lengths', 'Test Game Lengths', 'test_game_lengths', 100),
        ]

sub_dir_settings = [
        [('no_ladder', 'no ladder: 2x20')],
        [('ladder', 'ladder: 2x20')],
        ]

f_suffix = '0o0005'
save_dir = 'graphs'

plt.figure()
for sub_dirs in sub_dir_settings:
    for file_prefix, desc, image_file_suffix, max_y_val in graphs:
        for sub_dir, n_hidden in sub_dirs:
            directory = os.path.join('models', sub_dir)
            files = os.listdir(directory)
            data_files = [f for f in files if f.startswith(file_prefix) and 'png' not in f and f_suffix in f]
            if len(data_files) > 0:
                data = list()
                for f in data_files:
                    vals = np.load(os.path.join(directory, f)).squeeze()
                    data.append(np.minimum(max_y_val, vals))
            print('Merging the results from %d experiments into one graph...' % len(data))
            data = np.mean(np.array(data), axis=0)
            plt.plot(range(1,20*(data.shape[0])+1, 20), data, label='%s' % n_hidden, alpha=0.7)
plt.draw()
plt.title('Online Q-Learning: %s (LR=%s)' % (desc, f_suffix.replace('o', '.')))
plt.xlabel('Episode')
plt.ylabel(desc)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, '%s_%s.png' % (sub_dir, image_file_suffix)))

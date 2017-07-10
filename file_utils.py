import os


def lr_filename(directory, prefix, learning_rate, suffix=None):
    if suffix is not None:
        f = '%s_%s.%s' % (prefix, '{0:g}'.format(learning_rate).replace('.', 'o'), suffix)
    else:
        f = '%s_%s' % (prefix, '{0:g}'.format(learning_rate).replace('.', 'o'))
    return os.path.join(directory, f)

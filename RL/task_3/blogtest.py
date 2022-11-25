import numpy as np


def random_argmax(value_list):
    """ a random tie-breaking argmax """
    values = np.asarray(value_list)
    random_part = np.random.random(values.shape)
    equalizer = values == values.max()
    print(f"{random_part} * {equalizer} = {random_part * equalizer}")
    return np.argmax(random_part * equalizer)


result = random_argmax([5, 1, 3, 4, 5])
print(result)

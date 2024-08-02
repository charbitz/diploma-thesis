import numpy as np
from torch.utils.data.sampler import Sampler


class TomoBatchSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        batch_size (int): number of cases in a batch
        data_frame (DataFrame): data frame with views
    """

    def __init__(self, batch_size, data_frame, rng_ob):
        assert batch_size % 2 == 0
        self.batch_size = batch_size
        self.data_frame = data_frame
        self.positive = data_frame[(data_frame["Benign"] == 1) | (data_frame["Cancer"] == 1)]
        self.negative = data_frame[(data_frame["Normal"] == 1) | (data_frame["Actionable"] == 1)]
        self.rng_ob = rng_ob

    def __iter__(self):
        batch = []
        batch_index = 0
        for _ in range(len(self.positive) // 2):
            index_pos = self.positive.sample(random_state=self.rng_ob).index
            batch.append(self.data_frame.index.get_loc(index_pos[0]))
            if len(self.negative) > 0:
                index_neg = self.negative.sample(random_state=self.rng_ob).index
                batch.append(self.data_frame.index.get_loc(index_neg[0]))
            else:
                index_pos = self.positive.sample(random_state=self.rng_ob).index
                batch.append(self.data_frame.index.get_loc(index_pos[0]))
            if len(batch) >= self.batch_size:
                batch_index +=1
                yield batch
                batch = []

    def __len__(self):
        return len(self.positive) // self.batch_size

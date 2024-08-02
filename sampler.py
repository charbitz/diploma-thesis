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
                # print("batch:", batch_index)
                # print(*batch)
                yield batch
                batch = []


        # here :

        #     may update the self.positive (because this is the sampling source),
        #     in order not to sample again the same indexes for the next (and different) batch.
        #     We want different indexes to be sampled in different batches (i think)
        #  sth like: self.positive.drop([index_pos[0]])

        # # reset index to avoid problems with sampling non-existing in
        # self.positive = self.positive.reset_index(drop=True)
        #
        # # # create a copy of positive dataframe:
        # positive_copy = self.positive.copy()
        #
        # batch = []
        # # for _ in range(len(self.positive) // 2):
        # for _ in range(len(positive_copy) // 2):
        #     self.positive = self.positive.reset_index(drop=True)
        #
        #     index_pos = self.positive.sample().index
        #     elem=self.positive.index.get_loc(index_pos[0])
        #     batch.append(self.positive.index.get_loc(index_pos[0]))
        #
        #     self.positive = self.positive.drop(elem, axis=0)
        #     if len(self.negative) > 0:
        #         self.negative = self.negative.reset_index(drop=True)
        #
        #         index_neg = self.negative.sample().index
        #         batch.append(self.positive.index.get_loc(index_neg[0]))
        #
        #         self.negative = self.negative.drop(elem, axis=0)
        #     else:
        #         self.positive = self.positive.reset_index(drop=True)
        #
        #         index_pos = self.positive.sample().index
        #         elem = self.positive.index.get_loc(index_pos[0])
        #         batch.append(self.positive.index.get_loc(index_pos[0]))
        #
        #         self.positive = self.positive.drop(elem, axis=0)
        #     if len(batch) >= self.batch_size:
        #         yield batch
        #         batch = []

    # we should also write code to making drop last = False because  :
    # batch_sampler (in our case TomoBatchSampler) is Mutually exclusive with batch_size, shuffle, sampler, and drop_last

    def __len__(self):
        return len(self.positive) // self.batch_size

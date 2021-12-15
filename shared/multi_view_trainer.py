import torch
import collections

from transformers import Trainer
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    SequentialDistributedSampler,
)
from torch.utils.data.sampler import SequentialSampler,RandomSampler

class MultiViewRandomSampler(RandomSampler):
    def __init__(
        self,
        data_source,
        num_views=6
    ):
      super().__init__(data_source)
      self.num_views = num_views
      if len(data_source) % self.num_views != 0:
        raise ValueError("The length of the data source should be divisble by the number of views!")
    
    def __iter__(self):
      n = len(self.data_source) // self.num_views
      shuffled = (torch.randperm(n) * self.num_views).tolist()
      
      for i in shuffled:
        next_nums = [i]
        for _ in range(1, self.num_views):
          next_nums.append(next_nums[-1] + 1)
        yield from next_nums

    def __len__(self) -> int:
        return len(self.data_source)
      

class MultiViewTrainer(Trainer):
    def __init__(
        self,
        model = None,
        args = None,
        data_collator = None,
        train_dataset = None,
        eval_dataset = None,
        tokenizer = None,
        model_init = None,
        compute_metrics = None,
        callbacks = None,
        optimizers = (None, None),
        num_views = 6,
    ):
      super().__init__(
          model,
          args,
          data_collator,
          train_dataset,
          eval_dataset,
          tokenizer,
          model_init,
          compute_metrics,
          callbacks,
          optimizers,
      )
      self.num_views = num_views
    
    def _get_train_sampler(self):
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
            self.train_dataset, collections.abc.Sized
        ):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.train_dataset, self.args.train_batch_size, lengths=lengths, model_input_name=model_input_name
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.train_dataset,
                    self.args.train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                )

        else:
            # If args world size is greater than 1, behavior is undefined for MultiView trainer
            if self.args.world_size <= 1:
                #print(self.train_dataset)
                return MultiViewRandomSampler(self.train_dataset, num_views=self.num_views)
                # return SequentialSampler(self.train_dataset)
                # return RandomSampler(self.train_dataset)
            elif (
                self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                and not self.args.dataloader_drop_last
            ):
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                )
            else:
                return SequentialDistributedSampler(
                    self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    batch_size=self.args.per_device_train_batch_size,
                )
                # return DistributedSampler(
                #     self.train_dataset, num_replicas=self.args.world_size, rank=self.args.process_index
                # )

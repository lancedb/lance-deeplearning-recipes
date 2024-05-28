# Based on https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
import lance
import os
import math
import functools
import torch
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

import numpy as np
from transformers import AutoModelForCausalLM, CONFIG_MAPPING


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def from_indices(dataset, indices):
    """Load the elements on given indices from the dataset"""
    chunk = dataset.take(indices).to_pylist()
    chunk = list(map(lambda x: x["input_ids"], chunk))
    return chunk


class LanceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path,
        block_size,
    ):
        # Load the lance dataset from the saved path
        self.ds = lance.dataset(dataset_path)
        self.block_size = block_size

        # Doing this so the sampler never asks for an index at the end of text
        self.length = self.ds.count_rows() - block_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Generate a window of indices starting from the current idx to idx+block_size
        and return the tokens at those indices
        """
        window = np.arange(idx, idx + self.block_size)
        sample = from_indices(self.ds, window)

        return {"input_ids": torch.tensor(sample), "labels": torch.tensor(sample)}


class LanceDistributedSampler(torch.utils.data.Sampler):
    """Distributed sampler for LanceDataset."""

    def __init__(self, dataset, rank, num_replicas, block_size=512, shuffle=True):
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.block_size = block_size
        self.num_samples = len(self.dataset)
        self.epoch = 0

        # Get number of samples for each replica
        self.samples_per_replica = math.ceil(self.num_samples / self.num_replicas)

        # Get the start and end indices for the current replica using the rank
        start_idx = self.rank * self.samples_per_replica
        end_idx = min((self.rank + 1) * self.samples_per_replica, self.num_samples)

        # Generate indices available for current replica to access (they will be block size apart) and shuffle them
        self.available_indices = list(range(start_idx, end_idx, self.block_size))

        if shuffle:
            np.random.shuffle(self.available_indices)

    def __iter__(self):
        yield from self.available_indices

    def __len__(self):
        return len(self.available_indices)

    def set_epoch(self, epoch):
        self.epoch = epoch


def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx != 0 and batch_idx % 100 == 0 and rank == 0:
            print(
                "Batch: {} \tLoss: {:.6f}".format(batch_idx, ddp_loss[0] / ddp_loss[1])
            )

        # Put both input_ids and labels to the device
        for k, v in batch.items():
            batch[k] = v.to(rank)

        optimizer.zero_grad()
        output = model(**batch)
        loss = output.loss

        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(batch)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print("Train Epoch: {} \tLoss: {:.6f}".format(epoch, ddp_loss[0] / ddp_loss[1]))


def fsdp_main(rank, world_size, args):
    setup(rank, world_size)

    dataset = LanceDataset("wikitext_100K.lance/", 256)
    sampler = LanceDistributedSampler(
        dataset, block_size=256, rank=rank, num_replicas=world_size
    )

    train_kwargs = {"batch_size": args.batch_size, "sampler": sampler}
    cuda_kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": False}
    train_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset, **train_kwargs)
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )
    torch.cuda.set_device(rank)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    config = CONFIG_MAPPING["gpt2"]()
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).to(rank)

    model = FSDP(
        model,
        auto_wrap_policy=my_auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=True),
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    init_start_event.record()
    for epoch in range(1, args.epochs + 1):
        train(
            args,
            model,
            rank,
            world_size,
            train_loader,
            optimizer,
            epoch,
            sampler=sampler,
        )
        scheduler.step()

    init_end_event.record()

    if rank == 0:
        print(
            f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec"
        )
        print(f"{model}")

    if args.save_model:
        # use a barrier to make sure training is done on all ranks
        dist.barrier()
        states = model.state_dict()
        if rank == 0:
            torch.save(states, "gpt2.pt")

    cleanup()

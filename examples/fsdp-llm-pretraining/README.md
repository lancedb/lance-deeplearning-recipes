## Fully Sharded Data Parallel LLM pre-training using Lance text dataset

### Overview
This is the distributed training version of the LLM-pretraining example. Much of the training code used here was adapted from PyTorch's official [FSDP tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html). The key thing to note for distributed training is that you must set the spawn method in order for Lance dataset to work with your training code.

```python
from multiprocessing import set_start_method
set_start_method("spawn")
```

This will ensure that Lance dataset doesn't get stuck when you run your distributed training scripts. More details [here](https://github.com/lancedb/lance/issues/2204).

We'll be using the `wikitext_100K.lance` dataset created in the [Creating text dataset for LLM pre-training](https://colab.research.google.com/github/lancedb/lance-deeplearning-recipes/blob/main/examples/wikitext-llm-dataset/wikitext-llm-dataset.ipynb) example to train a basic GPT2 model from scratch using 🤗 transformers on a 2x NVIDIA RTX 2080ti GPUs in FSDP mode. The wikitext dataset, is a collection of over 100 million tokens extracted from the set of verified good and featured articles on Wikipedia.

To run training, first ensure all the requirements are installed and then run:

```shell
python train.py
```
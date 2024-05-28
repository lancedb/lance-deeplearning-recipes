## LLM pre-training using Lance text dataset

### Overview
Using a Lance text dataset for pre-training / fine-tuning a Large Language model is straightforward and memory-efficient. We'll be using the `wikitext_100K.lance` dataset that we created in the [Creating text dataset for LLM pre-training](https://colab.research.google.com/github/lancedb/lance-deeplearning-recipes/blob/main/examples/wikitext-llm-dataset/wikitext-llm-dataset.ipynb) example to train a basic GPT2 model from scratch using 🤗 transformers on a 1x A100 GPU. The wikitext dataset, is a collection of over 100 million tokens extracted from the set of verified good and featured articles on Wikipedia.

### Code and Blog
<a href="https://colab.research.google.com/github/lancedb/lance-deeplearning-recipes/blob/main/examples/llm-pretraining/llm-pretraining.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
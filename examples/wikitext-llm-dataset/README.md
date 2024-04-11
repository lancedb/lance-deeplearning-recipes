## Creating text dataset for LLM pre-training

### Overview
In this example, we will be creating a dataset for LLM pre-training by taking a 100K subset of [`wikitext-103-raw-v1`](https://huggingface.co/datasets/wikitext) dataset, tokenizing it and saving it as a Lance dataset. This can be done for as many or as few data samples as you wish with little memory consumption!

The wikitext dataset, is a collection of over 100 million tokens extracted from the set of verified good and featured articles on Wikipedia.

### Code and Blog
Below are the links for both the Google Colab walkthrough as well as the blog.

<a href="https://colab.research.google.com/github/lancedb/lance-deeplearning-recipes/blob/main/examples/wikitext-llm-dataset/wikitext-llm-dataset.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> [![Ghost](https://img.shields.io/badge/ghost-000?style=for-the-badge&logo=ghost&logoColor=%23F7DF1E)](https://blog.lancedb.com/custom-dataset-for-llm-training-using-lance/)
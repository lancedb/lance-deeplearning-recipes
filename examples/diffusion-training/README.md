## Diffusion Training

### Overview
In this minimal example, we show how to train a diffusion model to generate images using the [huggan/wikiart](https://huggingface.co/datasets/huggan/wikiart) which was converted into Lance format and is hosted on [Kaggle](https://www.kaggle.com/datasets/heyytanay/huggan-wikiart-lance).

The training code itself was inspired by Huggingface's [Diffusion model training example](https://huggingface.co/docs/diffusers/en/tutorials/basic_training) with necessary changes to adapt the training for Lance dataset.

### Code
<a href="https://colab.research.google.com/github/lancedb/lance-deeplearning-recipes/blob/main/examples/diffusion-training/diffusion-training.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> 
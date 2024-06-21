## CLIP Training

### Overview
In this minimal example, we train a CLIP model for natural language-based image search using the [Flickr8k dataset](https://github.com/goodwillyoga/Flickr8k_dataset) which was converted into Lance format and is hosted on [Kaggle](https://www.kaggle.com/datasets/heyytanay/flickr-8k-lance).

The training code itself was inspired by Manan Goel's excellent [Implementing CLIP With PyTorch Lightning](https://wandb.ai/manan-goel/coco-clip/reports/Implementing-CLIP-With-PyTorch-Lightning--VmlldzoyMzg4Njk1) training example with necessary changes to adapt the training for Lance dataset and task.

### Code
<a href="https://colab.research.google.com/github/lancedb/lance-deeplearning-recipes/blob/main/examples/clip-training/clip-training.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> 
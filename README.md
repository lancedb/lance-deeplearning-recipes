# Lance Deep Learning - recipes
<br />
Dive into building Deep learning pipelines using Lance datasets!
This repository contains examples to help you use Lance datasets for your Deep learning projects.

- These are built using Lance, a free, open-source, columnar data format that **requires no setup**.

- High-performance random access: More than **1000x faster** than Parquet.

- Zero-copy, automatic versioning: manage versions of your data automatically, and reduce redundancy with zero-copy logic built-in.
![318060905-d284accb-24b9-4404-8605-56483160e579](https://github.com/lancedb/lance-deeplearning-recipes/assets/15766192/8b350bf9-726e-45b8-ba23-dc8f2043c8aa)

<br />
Join our community for support - <a href="https://discord.gg/zMM32dvNtd">Discord</a> •
<a href="https://twitter.com/lancedb">Twitter</a>

---
<h3> Why Lance </h3>
<b>Convinience</b> <br />
Lance columnar file format is designed for large scale DL workloads. Columnar format allows you to easily and efficiently manage complex and unstructred multi-modal datasets Updation, filtering and zero-copy versioning allow you to iterate faster on large datasets. It’s designed to be used with images, videos, 3D point clouds, audio and of course tabular data. It supports any POSIX file systems, and cloud storage like AWS S3 and Google Cloud Storage

<br /><b> Performance </b> <br />
Lance format supports fast read/writes making your training time data loading significantly faster.

## Dataset Examples
Examples on how to convert existing datasets to Lance format.

| Example &nbsp; | Scripts &nbsp; | Read The Blog!&nbsp; &nbsp; &nbsp; &nbsp;|
|-------- | ------------- | -------------   |
| [Creating text dataset for LLM pre-training](/examples/wikitext-llm-dataset/) | <a href="https://colab.research.google.com/github/lancedb/lance-deeplearning-recipes/blob/main/examples/wikitext-llm-dataset/wikitext-llm-dataset.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> | [![Ghost](https://img.shields.io/badge/ghost-000?style=for-the-badge&logo=ghost&logoColor=%23F7DF1E)](https://blog.lancedb.com/custom-dataset-for-llm-training-using-lance/)|
| [Creating Instruction dataset for LLM fine-tuning](/examples/alpaca-dataset/) | <a href="https://colab.research.google.com/github/lancedb/lance-deeplearning-recipes/blob/main/examples/alpaca-dataset/alpaca-dataset.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> |
| [Creating Image Captioning Dataset for Multi-Modal Model Training](/examples/flickr8k-dataset/) | <a href="https://colab.research.google.com/github/lancedb/lance-deeplearning-recipes/blob/main/examples/flickr8k-dataset/flickr8k-dataset.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> |


## Training Examples
Practical examples showcasing how to adapt your Lance dataset to popular deep learning projects. 

| Example &nbsp; | Notebook & Scripts &nbsp; |
|-------- | ------------- |
| [PEFT Supervised Fine-tuning of Gemma using Huggingface Trainer](/examples/sft-gemma-hindi/) | <a href="https://colab.research.google.com/github/lancedb/lance-deeplearning-recipes/blob/main/examples/sft-gemma-hindi/sft_gemma_hindi.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> |
| [LLM pre-training](/examples/llm-pretraining/) | <a href="https://colab.research.google.com/github/lancedb/lance-deeplearning-recipes/blob/main/examples/llm-pretraining/llm-pretraining.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> |
| [COCO Image segmentation](/examples/image-segmentation/) | <a href="https://colab.research.google.com/github/lancedb/lance-deeplearning-recipes/blob/main/examples/image-segmentation/image-segmentation.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> |
| [FSDP LLM pre-training](/examples/fsdp-llm-pretraining/) |
| [Wikiart Diffusion Training](/examples/diffusion-training/) | <a href="https://colab.research.google.com/github/lancedb/lance-deeplearning-recipes/blob/main/examples/diffusion-training/diffusion-training.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> |

## Contributing Examples
If you're working on some cool deep learning examples using Lance that you'd like to add to this repo, please open a PR!

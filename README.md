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

This repository is divided into 2 sections:
- [Examples](#examples) - Get right into the code with minimal introduction, aimed at getting you from an idea to PoC within minutes!
<!-- - [Tutorials](#tutorials) - A curated list of tutorials, blogs, Colabs and courses to give you a headstart in using Lance datasets for your Deep learning projects.  -->

## Examples
Practical examples showcasing how to adapt your Lance dataset to popular deep learning projects. 

Examples are available as:
* **Colab notebooks** - that builds the application is stages allowing you to investigate results at every intermediate stage.
* **Python scripts** - for cases where you'd like directly to use the file or snippets to integrate in your application

If you're looking for in-depth tutorial-like examples, checkout the [tutorials](#tutorials) section!

| Example &nbsp; | Notebook & Scripts &nbsp; | Read The Blog!&nbsp; &nbsp; &nbsp; &nbsp;|
|-------- | ------------- | -------------   |
| [PEFT Supervised Fine-tuning of Gemma using Huggingface Trainer](/examples/sft-gemma-hindi/) | <a href="https://colab.research.google.com/github/lancedb/lance-deeplearning-recipes/blob/main/examples/sft-gemma-hindi/sft_gemma_hindi.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> |
| [Creating text dataset for LLM pre-training](/examples/wikitext-llm-dataset/) | <a href="https://colab.research.google.com/github/lancedb/lance-deeplearning-recipes/blob/main/examples/wikitext-llm-dataset/wikitext-llm-dataset.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> | [![Ghost](https://img.shields.io/badge/ghost-000?style=for-the-badge&logo=ghost&logoColor=%23F7DF1E)](https://blog.lancedb.com/custom-dataset-for-llm-training-using-lance/)|


## Contributing Examples
If you're working on some cool deep learning examples using Lance that you'd like to add to this repo, please open a PR!

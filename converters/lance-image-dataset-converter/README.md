## Convert any Image dataset to Lance

### Overview
The `convert-any-image-dataset-to-lance.py` script serves as a versatile tool for transforming any Image Dataset into the Lance format, enabling seamless integration and analysis. It provides a straightforward solution for converting diverse image datasets into a standardized format for enhanced compatibility and ease of use.

To effortlessly convert your image dataset into Lance format, simply execute the following command in your terminal:

```python
python convert-any-image-dataset-to-lance.py --dataset /path/to/your/image_dataset_folder
```

This command will seamlessly generate separate Lance files for your training, testing, and validation data. You have the flexibility to tailor the schema and specify which subsets (training, testing, validation) should be included in the Lance files. Even if you possess only one subset (e.g., only training data), you can conveniently indicate which subset(s) you wish to incorporate.

For starters and effortless access to pre-formatted CINIC-10 and mini-ImageNet datasets in Lance format, you can refer to the following Lance Image Dataset links:

CINIC-10 Dataset: https://www.kaggle.com/datasets/vipulmaheshwarii/cinic-10-lance-dataset

mini-ImageNet Dataset: https://www.kaggle.com/datasets/vipulmaheshwarii/mini-imagenet-lance-dataset

### Code and Blog
If you really want to understand the gist of how we are converting an Image Dataset to it's Lance format, refer to the following blog  and referenced colab explaining each step in detail. 

<a href="https://colab.research.google.com/drive/12RjdHmp6m9_Lx7YMRiat4_fYWZ2g63gx?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> [![Ghost](https://img.shields.io/badge/ghost-000?style=for-the-badge&logo=ghost&logoColor=%23F7DF1E)](https://blog.lancedb.com/convert-any-image-dataset-to-lance/)



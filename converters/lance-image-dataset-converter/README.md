## Convert any Image dataset to Lance

### Overview
The `convert-any-image-dataset-to-lance.py` script can be used for transforming any Image Dataset into the Lance format. It provides a straightforward solution for converting diverse image datasets into a standardized Lance format. 

To effortlessly convert your image dataset into Lance format, simply execute the following command in your terminal:

```python
python convert-any-image-dataset-to-lance.py --dataset /path/to/your/image_dataset_folder
```

This command will seamlessly generate separate Lance files for your training, testing, and validation data. You have the flexibility to change the schema and specify which subsets (training, testing, validation) should be included in the Lance files. Even if you have only one subset (e.g., only training data), you can conveniently indicate which subset(s) you wish to incorporate by changing the variables. 

For starters and effortless access to pre-formatted CINIC-10 and mini-ImageNet datasets in Lance format, you can refer to the following Lance Image Dataset links:

CINIC-10 Dataset: https://www.kaggle.com/datasets/vipulmaheshwarii/cinic-10-lance-dataset

mini-ImageNet Dataset: https://www.kaggle.com/datasets/vipulmaheshwarii/mini-imagenet-lance-dataset

### Code and Blog
If you want to understand the process of converting an image dataset to Lance format, please refer to the following blog and it's corresponding Google Colab notebook.

<a href="https://colab.research.google.com/drive/12RjdHmp6m9_Lx7YMRiat4_fYWZ2g63gx?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> [![Ghost](https://img.shields.io/badge/ghost-000?style=for-the-badge&logo=ghost&logoColor=%23F7DF1E)](https://blog.lancedb.com/convert-any-image-dataset-to-lance/)



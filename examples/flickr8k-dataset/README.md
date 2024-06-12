## Creating Image Captioning Dataset for Multi-Modal Model Training

### Overview
In this example, we will be creating an Image-caption pair dataset for Multi-modal model training by using the [Flickr8k_dataset](https://github.com/goodwillyoga/Flickr8k_dataset) and saving it in form of a Lance dataset with image file names, all captions for every image (order preserved) and the image itself (in binary format).

Flickr8k is a new benchmark collection for sentence-based image description and search, consisting of 8,000 images that are each paired with five different captions which provide clear descriptions of the salient entities and events. The images were chosen from six different Flickr groups, and tend not to contain any well-known people or locations, but were manually selected to depict a variety of scenes and situations

### Code and Blog
Below is the link for the Google Colab walkthrough.

<a href="https://colab.research.google.com/github/lancedb/lance-deeplearning-recipes/blob/main/examples/flickr8k-dataset/flickr8k-dataset.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
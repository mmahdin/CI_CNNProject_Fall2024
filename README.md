# Deep Learning for Image Processing and Captioning  

## Project Overview  
This project explores the application of computational intelligence techniques in computer vision through three distinct phases. Each phase introduces new challenges and tasks related to image classification, medical image analysis, and image captioning.

---

## Phase 1: Image Classification with Deep Neural Networks  

In this phase, a deep neural network (DNN) is implemented to classify images from the CIFAR-10 dataset. The dataset consists of 60,000 32x32 color images across 10 categories, which presents challenges due to low resolution and visual complexity.  

### Key Details:
- **Architecture**: Custom implementation of ResNet from scratch in a modular manner.
- **Optimization**: Adjusted the number of blocks in each stage and the number of stages to improve accuracy.
- **Results**: Achieved **94.29% accuracy** on the test set.
![result](https://i.imghippo.com/files/WYA2157aMs.png)
---

## Phase 2: Medical Image Analysis with CNNs  

This phase involves using convolutional neural networks (CNNs) to analyze breast histopathology images for cancer detection. The goal is to explore classification effectiveness and address data challenges.

### Key Details:
- **Data Analysis**: Comprehensive Exploratory Data Analysis (EDA) to understand data distribution and identify challenges.
- **Architectures**: Experimented with various ResNet architectures and implemented strong data augmentation to combat overfitting and address class imbalance.
- **Results**: Achieved an **F1 score of 90.02%** on the test set.
![saliency](https://i.imghippo.com/files/ari7503oVw.png)
---

## Phase 3: Image Captioning with Deep Learning Models  

In the final phase, the task is to generate descriptive captions for images using a combination of CNNs for feature extraction and RNNs for text generation.

### Key Details:
- **Architectures Explored**: RNN, LSTM, Attention LSTM, self-attention mechanisms, and multi-head self-attention.
- **Feature Extraction**: Utilized CNN models such as MobileNet, ResNet50, and EfficientNetB0.
- **Embeddings**: Implemented BERT and GloVe embeddings.
- **Reward Strategies**: Used BLEU scores for reward-based training.
- **Datasets**: Trained models on Flickr8k, Flickr30k, and 40k images from the MSCOCO dataset.
- **Implementation**: All modules, including attention mechanisms and LSTM models, were implemented from scratch.
- **Results Tracking**: Comprehensive trial and error process documented in commit history.
![1.jpg](https://i.postimg.cc/44zQHTTT/1.jpg)
![11.jpg](https://i.postimg.cc/Y9FZmyty/11.jpg)
![4.jpg](https://i.postimg.cc/nzqSLj1y/4.jpg)
![44.jpg](https://i.postimg.cc/d3BW8bpn/44.jpg)
![2.jpg](https://i.postimg.cc/c4kh0BSK/2.jpg)
![3.jpg](https://i.postimg.cc/J06N7M9w/3.jpg)
![5.png](https://i.postimg.cc/Y98g8t24/5.png)
---


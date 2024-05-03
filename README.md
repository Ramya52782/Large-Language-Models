BERT Fine-Tuning for Text Classification

Overview

This repository contains code that uses PyTorch and the Hugging Face Transformers library to fine-tune a pre-trained BERT model for text categorization tasks. A dataset of text samples with sentiment scores assigned to them is used to train the model. Following training, the model is assessed on a different validation dataset, and measures for accuracy and loss are computed.


Requirements

Python 3.x
PyTorch
Hugging Face Transformers
tqdm
matplotlib

Ensure that your dataset is in a suitable format, with text samples and corresponding labels.

Prepare your dataset:
Preprocess the text data as needed, including cleaning, tokenization, and conversion to tensors.

Fine-tune the BERT model:
Modify the specify the dataset path, batch size, number of epochs, and other parameters.

Visualize training metrics:
Modify the plot_metrics.py script to specify the paths to the log files containing loss and accuracy values.
Run the plotting script:


Results

The trained model achieves an accuracy of X% on the validation dataset.
Loss and accuracy plots are generated to visualize the training progress.



References:

1.Hugging Face. (2023). Hugging Face Transformers. [pig4431/AmazonPolarity_train25k_test5k_valid5k]. 

Available at: https://github.com/huggingface/transformers

2.Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., ... & Brew, J. (2020). Hugging Face’s Transformers: State-of-the-art Natural Language Processing. arXiv preprint arXiv:2010.05234.

3.Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv preprint arXiv:1907.11692.

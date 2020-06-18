# Abstractive Summarisation with Constrained Target Vocabularies

## Training Vocabulary Predictor on WikiCatsum Film dataset 
The dynamic vocavbulary predictor taken from the NLG-RL source code is modified to train long texts in our film_bow and film_raw dataset. The parameters set as following:

- Dimensionality for embeddings and hidden states: 256
- Maximum number of training epochs: 20
- Learning rate for AdaGrad: 8.0e-02
- Learning rate decay for AdaGrad: 0.5
- Batch size: 128
- Dropout rate for residual block: 0.4
- Weight decay rate for internal weight matrices: 1.0e-06
- Clipping value for gradient norm: 1.0

The training is performed with different values of small vocabulary size for evaluating recall (K). The results for BOW is found as follows.

|   K   | Recal Score |
| ----- | ----------- |
|   1   |   11.04 %   |
| 1000  |   89.92 %   |
| 2000  |   94.69 %   |
| 3000  |   96.72 %   |
| 4000  |   97.77 %   |
| 5000  |   98.46 %   |
| 6000  |   98.89 %   |


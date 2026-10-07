# Remaining useful life prediction on C-MAPSS

Project in the Deep Machine Learning course at Chalmers, by Hooman Hamze and
Benjamin Houshmand.

The task is to predict how many cycles a turbofan engine has left before it
fails, using the NASA C-MAPSS simulation data. We train one model on all four
sub-datasets (FD001-FD004) together, which mixes six operating conditions and
two fault modes.

## Model

- Input: a window of the last 50 cycles of one engine.
- The 21 sensor signals go through a 1D CNN (6 conv layers, batch norm,
  dropout, global average pooling) -> 64 features.
- The 3 operating settings go through an LSTM -> 64 features.
- Both are concatenated (the CNN part has a learnable scale) and passed
  through two linear layers to one RUL value.

Loss: smooth L1 plus 0.1 x a clamped version of the NASA score. The NASA score
penalises late predictions (predicting more life than is left) harder than
early ones, so it is added to push the model towards the safe side.

Training: Adam (lr 1e-3, weight decay 1e-4), ReduceLROnPlateau, gradient
clipping at 1.0, 15 epochs. Engines (not windows) are split into train and
validation, and the min-max scaling is fitted on the training engines only.

The notebook keeps the outputs of our run, including the validation
predictions after each epoch.

## Data

Download the C-MAPSS turbofan data set from the
[NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
and put the text files in `data/`:

```
data/train_FD001.txt  data/test_FD001.txt  data/RUL_FD001.txt
...
data/train_FD004.txt  data/test_FD004.txt  data/RUL_FD004.txt
```

## Running

```bash
pip install -r requirements.txt
jupyter notebook rul_cnn_lstm.ipynb
```

A GPU is used if available; on CPU an epoch takes a while because the
dataset is rebuilt window by window.

Reference: A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage propagation
modeling for aircraft engine run-to-failure simulation", PHM 2008.

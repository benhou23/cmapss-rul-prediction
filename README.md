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

Loss: smooth L1 plus 0.1 x a clamped version of the NASA score, which
penalises late predictions (more life predicted than is left) harder than
early ones.

Training: Adam (lr 1e-3, weight decay 1e-4), ReduceLROnPlateau, gradient
clipping at 1.0, 15 epochs, seed 42. The target is the RUL capped at 125
cycles, the usual convention for C-MAPSS. 10% of the training engines (not
windows) are held out for validation, and the min-max scaling is fitted on the
training engines only.

## Results

Standard protocol: one prediction per test engine, from its last 50 cycles,
against the true RUL in `RUL_FD00x.txt` (capped at 125). One model, trained
once on all four sub-datasets, on CPU (about 7 minutes).

| Test set | Engines | RMSE  | NASA score | Late predictions |
|----------|--------:|------:|-----------:|-----------------:|
| FD001    | 100     | 15.16 | 457        | 60%              |
| FD002    | 259     | 14.24 | 1069       | 54%              |
| FD003    | 100     | 15.69 | 1303       | 63%              |
| FD004    | 248     | 16.34 | 1596       | 47%              |

The NASA term in the loss did not make the model cautious: about half of the
test predictions are late, and near the end of life (true RUL below about 30)
the model tends to predict a few cycles too many. With weight 0.1 and the
clamp, the term is too weak to shift the predictions. A larger weight, or an
explicit bias towards early predictions, would be the next thing to try.

These numbers come from a single training run; expect some variation with
other seeds.

## Changes from the course version

The notebook was cleaned up after the course, and the results above come
from the cleaned-up version:

- Unit numbers restart in every sub-dataset, so engines are now identified by
  sub-dataset and unit. The course version grouped by unit number alone,
  which merged different engines into one series.
- The test sets were loaded but never evaluated; the evaluation above is new.
- RUL capped at 125, and a 10% validation split instead of 1%.
- Windows are built from per-engine arrays, which makes an epoch much faster
  on CPU.

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

A GPU is used if available; on a laptop CPU the full run takes about 10
minutes.

Reference: A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage propagation
modeling for aircraft engine run-to-failure simulation", PHM 2008.

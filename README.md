# Iris Classifier

Classifies iris flowers into three species from four petal and sepal measurements,
using a support vector machine from scikit-learn.

## Run it

```bash
pip install -r requirements.txt
python iris_svm.py
```

Output:

```
Training samples: 120
Test samples:     30
Test accuracy:    1.000
```

## What it does

The Iris dataset is 150 samples across three species — *setosa*, *versicolor* and
*virginica* — each measured for sepal length, sepal width, petal length and petal
width.

The script splits the data 80/20 into training and test sets, fits an `SVC` with
default settings (RBF kernel), and scores it on the held-out 30 samples it never
saw during training.

## About that accuracy

100% on the test set is normal here and not a sign of anything clever. Iris is a
small, clean, well-separated dataset — *setosa* is linearly separable from the other
two on petal measurements alone, and the remaining pair separate well enough that
most classifiers score in the high nineties.

`random_state=42` fixes the split. Without it, accuracy moves by a few points
between runs purely because different samples land in the test set, which makes the
number impossible to compare against anything.

## Files

```
iris_svm.py        the classifier
iris.csv           150 samples, 4 features plus species label
requirements.txt   pandas, scikit-learn
```

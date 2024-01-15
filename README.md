# One-Class_classification

Authors: Rafał Surdej, <a href="https://github.com/radamp11" target="_blank">Adam Stec</a>

One class classification model that generates outliers (data instances that deviate from normal ones)
and performs regular binary classification with standard scikit-learn binary classifiers.

## Model usage

```python
import numpy as np
from model.OneClassGen import OneClassGenerativeModel

X_train = np.array([[1,2], [1,3]])
model = OneClassGenerativeModel().fit(X_train)

X_test = np.array([[1,2]])

print(model.predict(X_test))
print(model.predict_proba(X_test))
```
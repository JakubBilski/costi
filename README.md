# COSTI

COSTI performs classification of sequences of temporal intervals, **with** or **without** intensity values.

Example sequence form Musekey dataset:

<p align="center">
<img src="https://raw.githubusercontent.com/JakubBilski/costi/main/assets/musekey_example.png"
  alt="Visualization of example sequence in intervals format: a piece played on a piano."
  width="686" height="289">
</p>

(intensity value is velocity of a key press).

You can use COSTI when your data can be expressed as:

sequence | channel | start | end | intensity
------- | ------- | ------- | ----- | ---------------- | 
*int <0-inf>* | *int <1-inf>* | *float* | *float* | *float, optional*
0 | 1 | 20.3 | 20.7 | 122.3 
0 | 1 | 20.7 | 65.4 | 66 
0 | 2 | 15 | 40.2 | 0.3
1 | 3 | 0 | 28.3 | 2736.3 
...

Events (rows) for the same sequence in the same channel should not overlap. All sequences are treated as if they started from zero. Data **does not** need to be sorted or normalized.

# Usage example
```python
import time
import numpy as np
from sklearn.linear_model import RidgeClassifierCV

import costi
import load_data
from train_test_split import cross_validate
from transform_to_input_format import transform_to_input_format


if __name__ == '__main__':

    x, y = load_data.load_musekey_unconstrained()

    cv_folds = cross_validate(x, y, 10)
    x_train, y_train, x_test, y_test = cv_folds[0]

    start_time = time.time()

    train_timestamps, train_channels, train_values, train_examples_s = transform_to_input_format(x_train)
    test_timestamps, test_channels, test_values, test_examples_s = transform_to_input_format(x_test)
    
    max_sequence_duration = max(np.max(train_timestamps), np.max(test_timestamps))

    x_train_transformed, parameters = costi.fit_and_transform_train(
        train_timestamps, train_channels, train_values, train_examples_s, max_sequence_duration)

    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
    classifier.fit(x_train_transformed, y_train)

    x_test_transformed = costi.transform_test(
        test_timestamps, test_channels, test_values, test_examples_s, parameters)

    score = classifier.score(x_test_transformed, y_test)
    elapsed_time = time.time()-start_time

    print(f"Accuracy: {score}")
    print(f"Time: {elapsed_time}")
```

# COSTI

COSTI performs classification of sequences of temporal intervals, **with** or **without** intensity values.


<p align="center">
<img src="../main/musekey_example.png"
  alt="Visualization of example sequence in intervals format: a piece played on a piano.">
  <i>Example sequence from Musekey dataset (intensity value = velocity of a key press)</i><br>
</p>


You can use COSTI when your data **could** be expressed as:

sequence | channel | start | end | intensity
------- | ------- | ------- | ----- | ---------------- | 
*int <0-inf>* | *int <1-inf>* | *float* | *float* | *float, optional*
0 | 1 | 20.3 | 20.7 | 122.3 
0 | 1 | 20.7 | 65.4 | 66 
0 | 2 | 15 | 40.2 | 0.3
1 | 3 | 0 | 28.3 | 2736.3 
...

Events for the same sequence in the same channel should not overlap. All sequences are treated as if they started from zero. Data **does not** need to be sorted or normalized.

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

# Input format

For example input files, see [data](../main/data). Data from files can be loaded with [load_data.py](../main/src/load_data.py), and then transformed with [transform_to_input_format.py](../main/src/transform_to_input_format.py), like in the example above.
Alternatively, you can create
```
train_timestamps, train_channels, train_values, train_examples_s
test_timestamps, test_channels, test_values, test_examples_s
```
on your own. Please note that these are 1-d vectors, so the information about sequences is concatenated. `train_timestamps` is all timestamps where any value changes (an event starts **or** ends). `train_channels` is a number of a channel (counting from 0) where this change happens. `train_values` is the value of the change. So, if an event with intensity value equal to 15 *begins*, the value will be 15, but if it *ends*, it will be -15. If your data does not contain intensity values, set all values to 1 and -1, depending on whether the timestamps is of event start or of event end. Finally, `train_examples_s` is a sequence of indexes where information about i-th sequence starts. So, `train_examples_s[0]` is always equal to zero (start index of the first sequence), `train_examples_s[1]` is the index where data for the second sequence begins and `train_examples_s[n]`=`train_timestamps.shape[0]`=`train_channels.shape[0]`=`train_values.shape[0]`.

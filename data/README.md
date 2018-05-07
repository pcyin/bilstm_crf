## HW4

This archive contains the following files:
  conlleval:                   evaluation script
  scores.py:                   python wrapper over conlleval
  train.data:                  CoNLL training data 
  dev.data:                    held out validation set
  train.data.half              approximately half of training data for debugging
  dev.data.half                approximately half of dev data for debugging
  train.data.quad              approximately quarter of training data for debugging
  dev.data.quad		       approximately quarter of dev data for debugging     
  README.txt:                  this file

NER tags are in BIO format, just like HW1. You may refer to
HW1's writeup in case there is any confusion.

To run the evaluation script while training, save the output
in a file and use `scores.py` to calculate the scores.
The eval script assumes that correct tags are in
the fourth column, and the output tags are in the fifth column. 

```python
from scores import scores
accuracy, precision, recall, f1 = scores(path2outputfile)
```

OR

To run the evaluation script (on a Unix system):
```sh
./conlleval < sample.output
```

Good luck!

# Credit Card Fraud Detection using SSVDD

This is a part of the Credit card fraud detection research project, available [HERE](https://arxiv.org/abs/2309.14880). In the project, different variants of many one class classification models were implemented. This code is an initial version of the SSVDD model used for the study mentioned before. It also serves as a demo of the python implementation of [SSVDD](https://github.com/Zaffarr/SSVDD_Python) model.

## Requirements

SSVDD model is implmented on top of the SVDD model, for which we have used the openly available python implementation of [SVDD](https://github.com/iqiukp/SVDD-Python/blob/master/src/BaseSVDD.py). Therefore, add this to the directory before implementing SSVDD.

## Citations

Please cite the following papers for using any part of this code.

```
@article{zaffar2023credit,
  title={Credit Card Fraud Detection with Subspace Learning-based One-Class Classification},
  author={Zaffar, Zaffar and Sohrab, Fahad and Kanniainen, Juho and Gabbouj, Moncef},
  journal={arXiv preprint arXiv:2309.14880},
  year={2023}
}

@inproceedings{sohrab2018subspace,
  title={Subspace support vector data description},
  author={Sohrab, Fahad and Raitoharju, Jenni and Gabbouj, Moncef and Iosifidis, Alexandros},
  booktitle={2018 24th International Conference on Pattern Recognition (ICPR)},
  pages={722--727},
  year={2018},
  organization={IEEE}
}
```


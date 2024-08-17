# Improving Cross-lingual Representation for Semantic Retrieval with Code-switching

Please cite:
```
@article{sr-codemix,
  author       = {Mieradilijiang Maimaiti and
                  Yuanhang Zheng and
                  Ji Zhang and
                  Fei Huang and
                  Yue Zhang and
                  Wenpei Luo and
                  Kaiyu Huang},
  title        = {Improving Cross-lingual Representation for Semantic Retrieval with
                  Code-switching},
  journal      = {arXiv preprint arXiv: 2403.01364},
  year         = {2024},
}
```

## Prepare the Code-Switched Data

1. Prepare the query and label data (e.g. `raw_data/Tatoeba.de-en.en` and `raw_data/Tatoeba.de-en.de`)

Both the query and label files should contain a series of sentences. One sentence per line.

Example:
`raw_data/Tatoeba.de-en.en`:
```
Let 's try something .
What is it ?
Today is June 18th and it is Muiriel 's birthday !
...
```

`raw_data/Tatoeba.de-en.de`:
```
Lass uns etwas versuchen !
Was ist das ?
Heute ist der 18. Juni und das ist der Geburtstag von Muiriel !
...
```

2. Prepare the dictionary data (e.g. `raw_data/en-de.txt` and `raw_data/de-en.txt`)

The dictionary data should conform the format of [MUSE](https://github.com/facebookresearch/MUSE).

You can directly download the dictionary from [MUSE](https://github.com/facebookresearch/MUSE), or prepare your own dictionary. For example, you may also prepare a dictionary using [ConceptNet](https://github.com/commonsense/conceptnet5/).

Example:
`raw_data/en-de.txt`:
```
the die
the der
the dem
the den
the das
and sowie
and und
was war
...
```

3. Run the Python script `codemix.py`

## Continual Pre-Training

1. Download the pre-trained language model from huggingface.
2. Run the script `train.sh` in the directory `cntptm`.

## Fine-Tuning

1. Run the script `train.sh` in the directory `ft` to train the model.
2. Run the script `predict.sh` to obtain the vector representation of the test query and label files.

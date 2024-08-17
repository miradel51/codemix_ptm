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

## Continual Pre-Training

1. Download the pre-trained language model from huggingface.
2. Run the script `train.sh` in the directory `cntptm`.

## Fine-Tuning

1. Run the script `train.sh` in the directory `ft` to train the model.
2. Run the script `predict.sh` to obtain the vector representation of the test query and label files.

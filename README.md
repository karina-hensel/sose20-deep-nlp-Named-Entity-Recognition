# Named Entity Recognition using a character- and word-based BiLSTM

This semester project implements a Bidirectional LSTM tagger (without the CRF layer) for named entity recognition as described in 'Lample et al. Neural Architectures for Named Entity Recognition. NAACL-HLT 2016.'. The results are evaluated on the CoNLL 2003 dataset and compared to the scores presented in the paper.


Due to the file size limit on GitHub the trained models can be downloaded from the following Kaggle repository: https://www.kaggle.com/karinahensel/ner-trainedlstms

To verify the results run the training script with the following command (with NUMBER_EPOCHS=50, DROPOUT=0.5) to generate the trained models:<br/>
<br/>
```python3 train.py PATH-TO-EMBEDDINGS-FILE PATH-TO-CONLL-DATASET NUMBER-EPOCHS DROPOUT MODEL-FILE-NAME```

Then run the evaluation script:<br/>
<br/>
```python3 evaluate.py PATH-TO-EMBEDDINGS-FILE PATH-TO-CONLL-DATASET MODEL-FILE-NAME```

# Toxic_Span_Detection
MSBD5018 NLP Group project

2022/04/02 Baseline BERT+CRF f1 score 66

2022/04/08 Baseline
- epoch=1, f1: 62.11
- epoch=2, f1: 65.2
- epoch=3, f1: 64.12


Pooling method = mean pooling
- Baseline + Fair Input (news-forward-fast)
    - epoch=1, f1: 64.92
    - epoch=2, f1: 59

- Baseline + Fair Input (news-backward-fast)
    - epoch=1, f1: 65.1 (use to ensemble)
    - epoch=2, f1: 58.93


Pooling method = sum pooling
- Baseline + Fair Input (news-forward-fast)
    - epoch=1, f1: 65.51 (use to ensemble)

- Baseline + Fair Input (news-backward-fast)
    - epoch=1, f1: 65.03


2022/04/13 

Pooling method = mean pooling
- Baseline + Fast text Input
    - epoch=1, f1: 66.46
    - epoch=2, f1: 62.27

- Baseline + Fast text Input + Fair Input (news-forward-fast)
    - epoch=1, f1: 65.13 (use to ensemble)

- Baseline + Fast text Input + Fair Input (news-backward-fast)
    - epoch=1, f1: 64.84 (use to ensemble)


Pooling method = sum pooling
- Baseline + Fast text Input
    - epoch=1, f1: 66.8 (use to ensemble)
    - epoch=2, f1: 63.9

- Baseline + Fast text Input + Fair Input (news-forward-fast)
    - epoch=1, f1: 63.84

- Baseline + Fast text Input + Fair Input (news-backward-fast)
    - epoch=1, f1: 63.93


2022/04/13 
#1 66.22 bert + epoch 3
#2 67.46 bert + lstm + epoch 3
#3 67.81 bert + lstm + 4_8_12 avg + epoch 2
#4 67.62 bert + lstm + 4_8_12 avg + pseudo_label + epoch 2
#E 68.14 vote ensemble(model#2 & model#3)
#E 68.18 intersection ensemble(model#1 & model#2 & model#3)
#E 68.34 intersection ensemble(model#2, model#3, flair, fasttext)

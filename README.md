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
    - epoch=1, f1: 65.1 
    - epoch=2, f1: 58.93


Pooling method = sum pooling
- Baseline + Fair Input (news-forward-fast)
    - epoch=1, f1: 65.51 (*)

- Baseline + Fair Input (news-backward-fast)
    - epoch=1, f1: 65.03

- Baseline + Fair Input + lstm (news-forward-fast)
    - epoch=2, f1: 55.71 (lstm: 896)
    - epoch=2, f1: 62.34 (lstm: 960)
    - epoch=2, f1: 63.31 (lstm: 1024)
    - epoch=2, f1: 63.74 (lstm: 1088)
    - epoch=2, f1: 66.81 (lstm: 1152) (*)
    - epoch=2, f1: 57.79 (lstm: 1280)

- Bert 4_8_12 avg + Fair Input + lstm (news-forward-fast)
    - epoch=2, f1: 65.24 (lstm: 1536)
    - epoch=2, f1: 65.79 (lstm: 1600)
    - epoch=2, f1: 67.64 (lstm: 1664) (*)
    - epoch=2, f1: 64.66 (lstm: 1728)
    - epoch=2, f1: 64 (lstm: 1792)
 

2022/04/22 

Pooling method = mean pooling
- Baseline + Fast text Input
    - epoch=1, f1: 66.46
    - epoch=2, f1: 62.27

- Baseline + Fast text Input + Fair Input (news-forward-fast)
    - epoch=1, f1: 65.13 

- Bert 4_8_12 avg + Fast text Input + Fair Input (news-forward-fast) 
    - epoch=2, f1: 66.51 (lstm: 1984) 
    - epoch=2, f1: 65.95 (lstm: 2016) 
    - epoch=2, f1: 66.65 (lstm: 2048) (*)
    - epoch=2, f1: 66.54 (lstm: 2080) 
    - epoch=2, f1: 64.47 (lstm: 2112)
    - epoch=2, f1: 62.81 (lstm: 2176)

- Baseline + Fast text Input + Fair Input (news-backward-fast)
    - epoch=1, f1: 64.84 


Pooling method = sum pooling
- Baseline + Fast text Input
    - epoch=1, f1: 66.8 (*)
    - epoch=2, f1: 63.9

- Baseline + Fast text Input + lstm
    - epoch=2, f1: 50.5（lstm: 256)
    - epoch=2, f1: 60.41（lstm: 352)
    - epoch=2, f1: 66.48 （lstm: 384) (*)
    - epoch=2, f1: 62.3（lstm: 448)
    - epoch=2, f1: 65.5 （lstm: 512)
    - epoch=2, f1: 66.17（lstm: 534)
    - epoch=2, f1: 60.07（lstm: 544)

- Bert 4_8_12 avg + Fast text Input + lstm
    - epoch=2, f1: 66.0（lstm: 1152)
    - epoch=2, f1: 67.51（lstm: 1216) (*)
    - epoch=2, f1: 66.77（lstm: 1280)
    - epoch=2, f1: 64.64（lstm: 1408)

- Baseline + Fast text Input + Fair Input (news-forward-fast)
    - epoch=1, f1: 63.84

- Baseline + Fast text Input + Fair Input (news-backward-fast)
    - epoch=1, f1: 63.93

- Bert 4_8_12 avg + Fast text Input + Fair Input (news-forward-fast)
    - epoch=2, f1: 64.39 (lstm: 1664)
    - epoch=2, f1: 63.07 (lstm: 1728)
    - epoch=2, f1: 67.21 (lstm: 1792) (*)
    - epoch=2, f1: 67.14 (lstm: 1984)
    - epoch=2, f1: 66.75 (lstm: 2048) 


2022/04/13 
#1 66.22 bert + epoch 3

#2 67.46 bert + lstm + epoch 3

#3 67.81 bert + lstm + 4_8_12 avg + epoch 2

#4 67.62 bert + lstm + 4_8_12 avg + pseudo_label + epoch 2

#E 68.14 vote ensemble(model#2 & model#3)

#E 68.18 intersection ensemble(model#1 & model#2 & model#3)

#E 68.34 intersection ensemble(model#2, model#3, flair, fasttext)

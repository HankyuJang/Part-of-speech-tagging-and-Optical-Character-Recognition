### Viterbi

How to handle words not in the trainset but appear in testset

- Set the emission prob to 0

```
==> So far scored 2000 sentences with 29442 words.
                   Words correct:     Sentences correct:
   0. Ground truth:      100.00%              100.00%
     1. Simplified:       18.60%                0.00%
         2. HMM VE:       18.60%                0.00%
        3. HMM MAP:       62.21%               30.15%
```

- Reproduce the algorithm by only removing the emission probability in calculation (Using the previous score and transition probability only)

```
==> So far scored 2000 sentences with 29442 words.
                   Words correct:     Sentences correct:
   0. Ground truth:      100.00%              100.00%
     1. Simplified:       18.60%                0.00%
         2. HMM VE:       18.60%                0.00%
        3. HMM MAP:       89.78%               31.55%
```


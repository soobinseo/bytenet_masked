# Bytenet with masking

A Machine Translation Tensorflow Implementation
Paper: [Neural Machine Translation in Linear Time](https://arxiv.org/abs/1610.10099)

## Notes
  * Few model structures are different from the paper
    * I used the IWSLT 2016 de-en dataset and the code to process the dataset has been changed slightly from the [original code of Kyubyung](https://github.com/Kyubyong/bytenet_translation)
    * I didn't implement 'Dynamic Unfolding'
    * I apply the masking for all residual blocks to eliminate the influence of pad embedding
    * I apply dropout just before the summation of residual block output.

## Requirements

  * Tensorflow >= 1.0.0
  * Numpy >= 1.11.1
  * nltk > 3.2.2

## Steps

1. Download [IWSLT 2016 German–English parallel corpus](https://wit3.fbk.eu/download.php?release=2016-01&type=texts&slang=de&tlang=en) and extract it to `data/` folder.
2. Run `train.py` with specific hyper parameters.
3. Run `translate.py` with same hyper parameters as above.

## Results
I got the Bleu Score 8.44 after 20 epochs. However, I got the Bleu score 44.69 by in-sampled data with embedding size 512, and I think it means that the model was trained well but overfitted. Therefore I suggest that you should try to run this model with larger dataset.
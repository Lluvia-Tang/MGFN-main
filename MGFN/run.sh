#!/bin/bash

# * laptop
# python ./train.py --model_name latgatsembert --dataset laptop --seed 1000 --bert_lr 2e-5 --num_epoch 15 --hidden_dim 768 --max_length 100 --vocab_dir ./dataset/Laptops_corenlp --device "cuda:0" --lex 8 --gamma 0.5 --theta 6

# * restaurants
# python ./train.py --model_name latgatsembert --dataset restaurant --seed 1000 --bert_lr 2e-5 --num_epoch 15 --vocab_dir ./dataset/Restaurants_corenlp --hidden_dim 768 --max_length 100 --device "cuda:0" --lex 0.06 --gamma 1 --theta 0.05

# * twitter
# python ./train.py --model_name latgatsembert --dataset twitter --seed 1000 --bert_lr 2e-5 --num_epoch 15 --vocab_dir ./dataset/Tweets_corenlp --hidden_dim 768 --max_length 100 --device "cuda:0" --lex 8 --gamma 0.5 --theta 6 



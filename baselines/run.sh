# nohup python -u train.py -gpu 2 -model LSTM -lr 0.003 -l2p 0 -lrdecay 0.9 -save_name lstm -data java --load_dataset > ./1.log &
nohup python -u train.py -gpu 0 -model Transformer -lr 4e-5 -save_name transformer3-4e-5 -data java --load_dataset > ./1.log &
# nohup python -u eval.py -gpu0 -model LSTM -save_name lstm/13.pt -data java --load_dataset > ./1.log &
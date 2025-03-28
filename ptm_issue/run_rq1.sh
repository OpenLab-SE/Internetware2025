DEVICE=0

# python train.py --model textcnn --embed none --device $DEVICE
# python train.py --model textcnn --embed glove --device $DEVICE
# python train.py --model textcnn --embed word2vec --device $DEVICE
# python train.py --model textcnn --embed fasttext --device $DEVICE

# python train.py --model bilstm --embed none --device $DEVICE
# python train.py --model bilstm --embed glove --device $DEVICE
# python train.py --model bilstm --embed word2vec --device $DEVICE
# python train.py --model bilstm --embed fasttext --device $DEVICE

# cdevice $DEVICE
# python train.py --model rcnn --embed glove --device $DEVICE
# python train.py --model rcnn --embed word2vec --device $DEVICE
# python train.py --model rcnn --embed fasttext --device $DEVICE

# python train.py --model 'bert-base-uncased' --embed none --device $DEVICE
# python train.py --model 'bert-base-uncased' --embed none --device $DEVICE --sequence
# python train.py --model 'albert-base-v2' --embed none --device $DEVICE
# python train.py --model 'roberta-base' --embed none --device $DEVICE  --sequence --disablefinetune --train_time 10
# python train.py --model 'huggingface/CodeBERTa-language-id' --embed none --device $DEVICE  --sequence --train_time 10
# python train.py --model 'xlnet-base-cased' --embed none  --train_time 1 --do_predict --device 1 --sequence
# python train.py --model '../models/seBERT/pytorch_model.bin' --embed none --device $DEVICE  --sequence --train_time 10 --local_model

# python train_new.py --train_file ./my_data/train/concat_concat/concat_concat.txt --test_file ./my_data/test/TTS/TTS.txt --model bert-base-uncased --embed none --sequence --train_time 1

# python train.py --model bert-base-uncased --embed none --train_time 1 --do_predict --device 1
# python train.py --model bert-base-uncased --embed none --sequence --train_time 1 --do_predict
# python train.py --model roberta-base --embed none --train_time 1 --do_predict
# python train.py --model albert-base-v2 --embed none --train_time 1 --do_predict
# python train.py --model xlnet-base-cased --embed none --sequence --train_time 1 --do_predict
# python train.py --model microsoft/codebert-base --embed none --train_time 1 --do_predict
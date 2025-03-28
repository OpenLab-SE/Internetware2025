DEVICE=0
TRAIN_TIME=1

FILE='./my_data/train/pytorch-CycleGAN-and-pix2pix_TRAIN_Aug/pytorch-CycleGAN-and-pix2pix_TRAIN_Aug.txt'
# python train.py --model bert-base-uncased --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE
# python train.py --model xlnet-base-cased --embed none  --train_time $TRAIN_TIME --do_predict --device $DEVICE --sequence --file $FILE
# python train.py --model roberta-base --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE
# python train.py --model albert-base-v2 --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE


FILE='./my_data/train/Real-Time-Voice-Cloning_TRAIN_Aug/Real-Time-Voice-Cloning_TRAIN_Aug.txt'
# python train.py --model bert-base-uncased --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE
python train.py --model xlnet-base-cased --embed none  --train_time $TRAIN_TIME --do_predict --device $DEVICE --sequence --file $FILE
# python train.py --model roberta-base --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE
# python train.py --model albert-base-v2 --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE


FILE='./my_data/train/EasyOCR_TRAIN_Aug/EasyOCR_TRAIN_Aug.txt'
# python train.py --model bert-base-uncased --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE
python train.py --model xlnet-base-cased --embed none  --train_time $TRAIN_TIME --do_predict --device $DEVICE --sequence --file $FILE
# python train.py --model roberta-base --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE
# python train.py --model albert-base-v2 --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE


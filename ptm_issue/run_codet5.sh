DEVICE=0
TRAIN_TIME=10


TRAIN_FILE='./my_data/train/framework_newlabel_clean_TRAIN_Aug/framework_newlabel_clean_TRAIN_Aug.txt'
VALID_FILE='./my_data/valid/framework_newlabel_clean_VALID_Aug/framework_newlabel_clean_VALID_Aug.txt'
TEST_FILE='./my_data/test/framework_newlabel_clean_TEST_Aug/framework_newlabel_clean_TEST_Aug.txt'

python train_cross.py --local_model \
    --model /workspace/issue_classify_1/code/models/codet5-base \
    --trial codet5_framework \
    --embed none \
    --train_time $TRAIN_TIME \
    --device $DEVICE \
    --train_file $TRAIN_FILE \
    --valid_file $VALID_FILE \
    --test_file $TEST_FILE \
    --do_predict \
    --batch_size 16 \
    --base_lr 2.5e-6 \
    --sequence

# python train_cross.py --local_model --model models/t5-large --embed none --train_time $TRAIN_TIME --device $DEVICE --train_file $TRAIN_FILE --test_file $TEST_FILE --sequence
# FILE='./my_data/train/streamlit1_TRAIN_Aug/streamlit1_TRAIN_Aug.txt'
# python train.py --model models/BERTOverflow --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE --local_model --sequence
# python train.py --model models/bert-base-uncased/pytorch_model.bin --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE --local_model
# python train.py --model bert-base-uncased --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE
# python train.py --model models/xlnet-base-cased --embed none  --train_time $TRAIN_TIME --do_predict --device $DEVICE --sequence --file $FILE --local_model
# python train.py --model xlnet-base-cased --embed none  --train_time $TRAIN_TIME --do_predict --device $DEVICE --sequence --file $FILE 
# python train.py --model models/roberta-base --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE --local_model --sequence
# python train.py --model roberta-base --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE
# python train.py --model models/albert-base-v2 --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE --local_model   
# python train.py --model albert-base-v2 --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE 

# python train.py --model textcnn --embed word2vec --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE
# python train.py --model bilstm --embed glove --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE
# python train.py --model rcnn --embed glove --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE

# FILE='./my_data/train/contact_TRAIN_Aug/contact_TRAIN_Aug.txt'
# python train.py --model bert-base-uncased --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE
# python train.py --model xlnet-base-cased --embed none  --train_time $TRAIN_TIME --do_predict --device $DEVICE --sequence --file $FILE

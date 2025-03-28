DEVICE=0
TRAIN_TIME=10


TRAIN_FILE='./my_data/train/framework_newlabel_clean_TRAIN_Aug/framework_newlabel_clean_TRAIN_Aug.txt'
VALID_FILE='./my_data/valid/framework_newlabel_clean_VALID_Aug/framework_newlabel_clean_VALID_Aug.txt'
TEST_FILE='./my_data/test/framework_newlabel_clean_TEST_Aug/framework_newlabel_clean_TEST_Aug.txt'

python train_cross.py --local_model \
    --model /workspace/issue_classify_1/code/models/codebert-base \
    --trial codebert_framework \
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
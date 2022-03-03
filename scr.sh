PREDICT_PATH=~/Stuff/Images/results/march3_multi/blurred
SAVE_PATH=~/Stuff/Images/results/temp
ORIGIN_W=~/Repos/DeblurGANv2/new_pretrained/l9_multi_last.h5
METHOD_W=~/Repos/DeblurGANv2/new_pretrained/l9_mono_last.h5

# base method
python3 predict.py "${PREDICT_PATH}/*" --weights_path=$ORIGIN_W --out_dir=$SAVE_PATH/base/

# proposed method
# 1) rotate
python3 rotate.py r $PREDICT_PATH $SAVE_PATH/rotated

# 2) transform w/ spezialized neural network
python3 predict.py "${SAVE_PATH}/rotated/*" --weights_path=$METHOD_W --out_dir="${SAVE_PATH}/restored"
# python3 predict.py '../Diploma-1/pict/b/*' --weights_path=./pretrained_models/best_fpn_l3.h5  --out_dir=../Diploma-1/pict/c/l03/
# python3 predict.py '../Diploma-1/pict/b/*' --weights_path=./pretrained_models/best_fpn_l5.h5  --out_dir=../Diploma-1/pict/c/l05/
# python3 predict.py '../Diploma-1/pict/b/*' --weights_path=./pretrained_models/best_fpn_l7.h5  --out_dir=../Diploma-1/pict/c/l07/
# python3 predict.py '../Diploma-1/pict/b/*' --weights_path=./pretrained_models/best_fpn_l9.h5  --out_dir=../Diploma-1/pict/c/l09/
# python3 predict.py '../Diploma-1/pict/b/*' --weights_path=./pretrained_models/best_fpn_l11.h5 --out_dir=../Diploma-1/pict/c/l11/


# 3) (very optional) restore from defocus
# python3 predict.py '../Diploma-1/pict/c/l03/*' --weights_path=./pretrained_models/fpn_inception.h5 --out_dir=../Diploma-1/pict/c/l03/da/
# python3 predict.py '../Diploma-1/pict/c/l05/*' --weights_path=./pretrained_models/fpn_inception.h5 --out_dir=../Diploma-1/pict/c/l05/da/
# python3 predict.py '../Diploma-1/pict/c/l07/*' --weights_path=./pretrained_models/fpn_inception.h5 --out_dir=../Diploma-1/pict/c/l07/da/
# python3 predict.py '../Diploma-1/pict/c/l09/*' --weights_path=./pretrained_models/fpn_inception.h5 --out_dir=../Diploma-1/pict/c/l09/da/
# python3 predict.py '../Diploma-1/pict/c/l11/*' --weights_path=./pretrained_models/fpn_inception.h5 --out_dir=../Diploma-1/pict/c/l11/da/

# python3 predict.py '../Diploma-1/pict/c/l09/*' --weights_path=./pretrained_models/fpn_inception.h5 --out_dir=../Diploma-1/pict/c/l09/da/

# 4) rotate back
python3 rotate.py rb "${SAVE_PATH}/restored" "${SAVE_PATH}/result"
# python3 rotate.py rb ./pict/c/l03/da ./pict/d/l03
# python3 rotate.py rb ./pict/c/l05/da ./pict/d/l05
# python3 rotate.py rb ./pict/c/l07/da ./pict/d/l07
# python3 rotate.py rb ./pict/c/l09/da ./pict/d/l09
# python3 rotate.py rb ./pict/c/l11/da ./pict/d/l11

import argparse
from Regression_ker import trainer
#from Speed_est import predicter
import os


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=r'v1', type=str)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--res_dir', default=r'.\Regression_Result\tmp1', type=str)
    parser.add_argument('--param_dir', default=r'.\param_tmp', type=str)
    parser.add_argument('--pretrain', default=False)
    parser.add_argument('--model_dir', default=r'.\model_param', type=str)

    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.002, type=float, help='Learning rate')
    parser.add_argument('--gamma', default=0.87, type=float)

    parser.add_argument('--data_root', default=r".\rotating_no_tur_filter_test")
    # opening_set = [1,2,3,4,5,6,8,10,12,15,18,20,30,36]
    parser.add_argument('--opening_set', default=[1,2,3,4,5,6,8,10,12,15,18,20,30,36])
    parser.add_argument('--et_set', default=[100,200,300,400,500,600,700,800,900,1000])
    parser.add_argument('--rs_set', default=[1,2,5,10,15,20,30,40,50,60])
    parser.add_argument('--voa_set', default=[10], type=int)
    parser.add_argument('--data_num', default=100, type=int, help='number of each subclass')

    return parser

if __name__ == '__main__':
    args = create_parser().parse_args()
    for open in args.opening_set:
        for voa in args.voa_set:
                for et in args.et_set:
                    img_name_set = []
                    data_dir = os.path.join(args.data_root, rf'{str(open)}\{str(voa)}\{str(et)}')
                    for rs in args.rs_set:
                            for i in range(1,args.data_num + 1):
                                pattern = rf'angle_{open}_{voa}db_{et}ms_{rs}hz_{i}.png'
                                img_name_set.append(pattern)

                    Trainer = trainer(args, data_dir, img_name_set, open, et, voa)







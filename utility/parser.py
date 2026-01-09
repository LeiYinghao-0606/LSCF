import argparse

def parse_args():
    parse = argparse.ArgumentParser(description="Run Reproduct")
    parse.add_argument('--seed', type=int, default=2024, help='random seed')
    parse.add_argument('--item_knn_k', type=int, default=200, help='K')
    parse.add_argument('--item_knn_max_degree', type=int, default=400, help='K')
    parse.add_argument('--gpu', type=int, default=0, help='indicates which gpu to use')
    parse.add_argument('--cuda', type=bool, default=True, help='use gpu or not')
    parse.add_argument('--log', type=str, default='None', help='init log file name')
    parse.add_argument('--dataset_path', type=str, default='./dataset/', help='choice dataset')
    parse.add_argument('--dataset_type', type=str, default='.txt', help='choice dataset')
    parse.add_argument('--dataset', type=str, default='amazon-book', help='choice dataset')
    parse.add_argument('--top_K', type=str, default='[10, 20, 30, 40, 50]')
    parse.add_argument('--train_epoch', type=int, default=600)
    parse.add_argument('--early_stop', type=int, default=15)
    parse.add_argument('--embedding_size', type=int, default=64)
    parse.add_argument('--train_batch_size', type=int, default=4096)
    parse.add_argument('--test_batch_size', type=int, default=4096)
    parse.add_argument('--learn_rate', type=float, default=0.001)
    parse.add_argument('--reg_lambda', type=float, default=0.0001)
    parse.add_argument('--gcn_layer', type=int, default=1)
    parse.add_argument('--test_frequency', type=int, default=1)
    parse.add_argument('--sparsity_test', type=int, default=0)
    parse.add_argument('--tau', type=float, default=0.22)
    parse.add_argument('--ssl_lambda', type=float, default=5.0)
    parse.add_argument('--encoder', type=str, default='SDCA')
    parse.add_argument('--neg_sample_mode', type=str, default='random')
    parse.add_argument('--sg_alpha', type=float, default=0.95,
                        help='douban : 0.88 /  amazon : 0.95  /tmall 0.94 ')
    parse.add_argument('--lap_lambda', type=float, default=0.0,
                       help='douban : 0.05 /  amazon : 0.1  /tmall 0.05 ')
    parse.add_argument('--na_neg_k', type=int, default=4096,
                       help='douban : 4096 /  amazon : 8192  /tmall 8192 ')
    parse.add_argument('--na_q_beta', type=float, default=0.5,
                        help='douban : 0.5 /  amazon : 0.5  /tmall 0.5 ')
    parse.add_argument('--na_mix_uniform', type=float, default=0.75,
                        help='douban : 0.7 /  amazon : 0.75  /tmall 0.75 ')
    
    
    



    return parse.parse_args()

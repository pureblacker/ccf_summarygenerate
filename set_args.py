import argparse
def set_args():
    parser=argparse.ArgumentParser(description="Arguments for training summary generation model!")
    parser.add_argument('--epoch', default= 3,type=int, action='store', help='Number of epochs to run')
    parser.add_argument('--warmup', default=300, type=int, action='store', help='Number of warmup steps to run')
    parser.add_argument('--model_name', default='mymodel.pt', type=str, action='store', help='Name of the model file')
    parser.add_argument('--data_file', default='mydata.csv', type=str, action='store', help='Name of the data file')
    parser.add_argument('--batch', type=int, default=32, action='store', help='Batch size')
    parser.add_argument('--learning_rate', default=3e-5, type=float, action='store', help='Learning rate for the model')
    parser.add_argument('--src_maxlen', default=200, type=int, action='store', help='Maximum length of srcsequence')
    parser.add_argument('--tar_maxlen', default=200, type=int, action='store', help='Maximum length of tarsequence')
    args = parser.parse_args()
    return args
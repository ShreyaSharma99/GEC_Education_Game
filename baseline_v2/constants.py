# datapath = '/Users/shreya/Desktop/GEC_ETH_Project/GEC_baseline/gec_nucle_new.csv'
datapath = '../gec_nucle_new.csv'

# 35 - nucle max replace/add len
# 21 - 10_gec replace/add max_len
MAX_INPUT_LEN = 256
MAX_TARGET_LEN = MAX_INPUT_LEN
# For 10_gec = 227

Actions = ["add", "replace", "delete"]
BATCH_SIZE =2

EPOCHS = 5

NROWS = 10
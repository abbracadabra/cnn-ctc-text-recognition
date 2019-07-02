import os

num_chars = 5991
epochs = 30000000
base_dir = os.getcwd()
image_dir = r'C:\迅雷下载\Synthetic Chinese String Dataset\images'
label_path = r'C:\迅雷下载\train.txt'
#label_path = os.path.join(base_dir,"testlb.txt")
batch_size = 10
log_dir = os.path.join(base_dir,'log')
model_dir = os.path.join(base_dir,'mdl','mdl')
char_map = os.path.join(base_dir,'chars')
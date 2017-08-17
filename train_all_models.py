from faceRecg import train_all_model
import sys

print(sys.argv)
train_all_model(epochs_num = int(sys.argv[1]),len_of_test = int(sys.argv[2]))

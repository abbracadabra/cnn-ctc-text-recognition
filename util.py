from config import *

def loadindex2char():
    f = open(char_map,encoding='utf-8')
    fl = f.readlines()
    mpp = {i:l.strip() for i,(l) in enumerate(fl)}
    return mpp

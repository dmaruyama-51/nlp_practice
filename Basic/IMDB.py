import glob 
import os 
import io 
import string 
import os
import urllib.request
import tarfile

def download_data():
    url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    save_path = "./data/aclImdb_v1.tar.gz"
    if not os.path.exists(save_path):
        urllib.request.urlretrieve(url, save_path)
    tar = tarfile.open('./data/aclImdb_v1.tar.gz')
    tar.extractall('./data/')  
    tar.close()  

if __name__ == "__main__":
    download_data()
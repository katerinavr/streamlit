import os
import wget

def download_file(local_path, link):

    if not os.path.exists(local_path):
        print('Downloading from %s, this may take a while...' % link)
        wget.download(link, local_path)
    
    return local_path
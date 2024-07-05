import logging
import os, os.path
import time

log_file_name = "log-"+time.strftime("%Y-%m-%d-%Hh%Mm%Ss", time.localtime())+".log"


if not os.path.exists("log/"):
    os.makedirs("log/")
if not os.path.exists("log/"+log_file_name):
    with open("log/"+log_file_name, mode='w', encoding='utf-8') as file:
        file.close()

logging.basicConfig(filename='log/'+log_file_name, encoding='utf-8', level=logging.INFO)
logging.info("--------------START--------------")
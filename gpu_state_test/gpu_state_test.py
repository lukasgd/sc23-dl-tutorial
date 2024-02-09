import torch
import socket
import os

gpu_id = int(os.environ['SLURM_LOCALID'])
device = "cuda:" + str(gpu_id)
try:
    torch.zeros(2).cuda(device)
    print(socket.gethostname() + ":" + device + " ==> OK ")
except Exception as e:
    print(socket.gethostname() + ":" + device + " ==> NOK " + str(e)[:120] + " ... ")

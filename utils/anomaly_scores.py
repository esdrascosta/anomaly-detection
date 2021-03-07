import torch
from utils.msgmsd_map import msgmsd
from kornia.filters.median import median_blur

def compute_gms_anomaly_score(recon, x):
    
    with torch.no_grad():
        msgms_map = msgmsd(recon, x)
        output_maps = median_blur(msgms_map, (21, 21))
        return output_maps.max()

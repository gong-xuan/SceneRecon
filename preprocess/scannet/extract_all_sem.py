import os, cv2
import numpy as np

LFW_PARTS_PALETTE = {
    0: (0, 0, 0) , 
    1: (0, 148, 218), 
    2: (90, 90, 90), 
    3: (52,124,49), 
    4: (52,52,255), 
    5: (52,52,137), 
    6: (0,255,255), 
    7: (143,71,143), 
    8: (255,255,255), 
    9: (119,43,43), 
    10: (165,148,44), 
    11: (255,0,0),
    12: (0,255, 0),
    13: (0,0,255),
    14: (255, 255, 0),
    15: (0, 255, 128),
    16: (255, 0, 255),
    17: (255, 0, 128),
    18: (128, 0, 255),
    19: (128, 255, 0),
    20: (255, 128, 0),
    21: (0, 128, 255)
}


def seglabel2color(seglabel):
    H, W = seglabel.shape
    segcolor = np.zeros((H,W,3))
    for label in np.unique(seglabel):
        segcolor[seglabel==label] = LFW_PARTS_PALETTE[label]
    return segcolor


def check_scene_label(scene_path):
    seg_path1 = f'{scene_path}/label'
    seg_path2 = f'{scene_path}/label-filt'
    img_path = f'{scene_path}/color'
    for imgf in os.listdir(img_path):
        img = cv2.imread(f'{img_path}/{imgf}')
        seg1 = cv2.imread(f'{seg_path1}/{imgf[:-4]}.png', cv2.IMREAD_GRAYSCALE)
        seg2 = cv2.imread(f'{seg_path2}/{imgf[:-4]}.png', cv2.IMREAD_GRAYSCALE)
        seg1 = seglabel2color(seg1)
        seg2 = seglabel2color(seg2)
        imgall = np.concatenate((img, seg1, seg2), axis=1)
        cv2.imwrite( f'debug/{imgf}.png', imgall)
        print("Complete..", imgf)
    import pdb; pdb.set_trace()




if __name__ == '__main__':
    scene_path = '/data_local/xuangong/scannet/scans/scene0000_00'
    check_scene_label(scene_path)

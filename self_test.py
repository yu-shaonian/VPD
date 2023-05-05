import glob
import os

ctag = glob.glob(os.path.join('/home/guojun/dataset/ScanNetv2/scans_test/scene0707_00/sensor_data',
                              'frame-*.color.jpg'))

for i in ctag:
    a = i.split('.')[0].split('-')[1]  #num
    img_name = '/home/guojun/dataset/ScanNetv2/scans_test/scene0707_00/sensor_data/' + f'frame-{a}.color.jpg'
    img_name_new = f'/home/guojun/dataset/ScanNetv2/scans_test/scene0707_00/new_scan/{a}.jpg'
    os.system(f'cp {img_name} {img_name_new}')
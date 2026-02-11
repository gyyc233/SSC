import numpy as np
from matplotlib import pyplot as plt
import sys
import os

# 用 KITTI 位姿 为指定序列生成正样本对和负样本对的索引

def run(seq='00',folder="/media/l/yp2/KITTI/odometry/dataset/poses/"):
    pose_file=os.path.join(folder,seq+".txt")
    poses=np.genfromtxt(pose_file)
    poses=poses[:,[3,11]] # 读取序列文件每帧的 x y

    # 计算帧之间的平面距离
    inner=2*np.matmul(poses,poses.T)
    xx=np.sum(poses**2,1,keepdims=True)
    dis=xx-inner+xx.T
    dis=np.sqrt(np.abs(dis))

    # 正负样本筛选
    id_pos=np.argwhere(dis<3)
    id_pos=id_pos[id_pos[:,0]-id_pos[:,1]>50] # 筛选出距离小于3米且帧号差大于50的帧对作为正样本

    id_neg=np.argwhere(dis>20)
    id_neg=id_neg[id_neg[:,0]>id_neg[:,1]] # 筛选出距离大于20米且帧号大的帧作为负样本
    print(len(id_pos))
    np.savez(seq+'.npz',pos=id_pos,neg=id_neg)
    # 生成的 .npz 需再写成 eval_seq 用的 pairs 文本（每行 sequ1 sequ2 label）：用 pos 写 label=1，用 neg 按比例采样写 label=0


if __name__=='__main__':
    seq="05"
    if len(sys.argv)>1:
        seq=sys.argv[1]
    run(seq,"/media/l/yp2/KITTI/odometry/dataset/poses/")

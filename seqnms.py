# -*- coding: utf-8 -*-
import _init_paths
import numpy as np
from fast_rcnn.nms_wrapper import nms
import cv2
import time
import copy
import cPickle as pickle
import os

CLASSES = ('__background__',
           'airplane','antelope','bear','bicycle','bird','bus',
           'car','cattle','dog','domestic cat','elephant','fox',
           'giant panda','hamster','horse','lion','lizard','monkey',
           'motorcycle','rabbit','red panda','sheep','snake','squirrel',
           'tiger','train','turtle','watercraft','whale','zebra')

CONF_THRESH = 0.5
NMS_THRESH = 0.3
IOU_THRESH = 0.6

'''
修改检测结果格式，用作后续处理
第一维：种类
第二维：帧
第三维：bbox
第四维：x1,y1,x2,y2,score
'''
def createInputs(video):
    create_begin=time.time()
    frames=sorted(video.keys()) #获得按序排列的帧的名称
    dets=[[] for i in CLASSES[1:]] #保存最终结果
    for cls_ind,cls in enumerate(CLASSES[1:]): #类
        for frame_ind,frame in enumerate(frames): #帧
            cls_boxes = video[frame]['boxes'][:, 4*(cls_ind+1):4*(cls_ind + 2)]
            cls_scores = video[frame]['scores'][:, cls_ind+1]
            cls_dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float64)
            dets[cls_ind].append(cls_dets)
    create_end=time.time()
    print 'create inputs: {:.4f}s'.format(create_end - create_begin)
    return dets

def createLinks(dets_all):
    links_all=[]
    #建立每相邻两帧之间的link关系
    frame_num=len(dets_all[0])
    cls_num=len(CLASSES)-1
    #links_all=[] #保存每一类的全部link，第一维为类数，第二维为帧数-1，为该类下的links即每一帧与后一帧之间的link，第三维每帧的box数，为该帧与后一帧之间的link
    for cls_ind in range(cls_num): #第一层循环，类数
        links_cls=[] #保存一类下全部帧的links
        link_begin=time.time()
        for frame_ind in range(frame_num-1): #第二层循环，帧数-1，不循环最后一帧
            dets1=dets_all[cls_ind][frame_ind]
            dets2=dets_all[cls_ind][frame_ind+1]
            box1_num=len(dets1)
            box2_num=len(dets2)
            #先计算每个box的area
            if frame_ind==0:
                areas1=np.empty(box1_num)
                for box1_ind,box1 in enumerate(dets1):
                    areas1[box1_ind]=(box1[2]-box1[0]+1)*(box1[3]-box1[1]+1)
            else: #当前帧的area1就是前一帧的area2，避免重复计算
                areas1=areas2
            areas2=np.empty(box2_num)
            for box2_ind,box2 in enumerate(dets2):
                areas2[box2_ind]=(box2[2]-box2[0]+1)*(box2[3]-box2[1]+1)
            #计算相邻两帧同一类的link
            links_frame=[] #保存相邻两帧的links
            for box1_ind,box1 in enumerate(dets1):
                area1=areas1[box1_ind]
                x1=np.maximum(box1[0],dets2[:,0])
                y1=np.maximum(box1[1],dets2[:,1])
                x2=np.minimum(box1[2],dets2[:,2])
                y2=np.minimum(box1[3],dets2[:,3])
                w =np.maximum(0.0, x2 - x1 + 1)
                h =np.maximum(0.0, y2 - y1 + 1)
                inter = w * h
                ovrs = inter / (area1 + areas2 - inter)
                links_box=[ovr_ind for ovr_ind,ovr in enumerate(ovrs) if ovr >= IOU_THRESH] #保存第一帧的一个box对第二帧全部box的link
                links_frame.append(links_box)
            links_cls.append(links_frame)
        link_end=time.time()
        print 'link: {:.4f}s'.format(link_end - link_begin)
        links_all.append(links_cls)
    return links_all

def maxPath(dets_all,links_all):
    for cls_ind,links_cls in enumerate(links_all):
        max_begin=time.time()
        dets_cls=dets_all[cls_ind]
        while True:
            rootindex,maxpath,maxsum=findMaxPath(links_cls,dets_cls)
            if len(maxpath) <= 1:
                break
            rescore(dets_cls,rootindex,maxpath,maxsum)
            deleteLink(dets_cls,links_cls,rootindex,maxpath,IOU_THRESH)
        max_end=time.time()
        print 'max path: {:.4f}s'.format(max_end - max_begin)

def NMS(dets_all):
    for cls_ind,dets_cls in enumerate(dets_all):
        for frame_ind,dets in enumerate(dets_cls):
            keep=nms(dets, NMS_THRESH)
            dets_all[cls_ind][frame_ind]=dets[keep, :]

def findMaxPath(links,dets):
    maxpaths=[] #保存从每个结点到最后的最大路径与分数
    roots=[] #保存所有的可作为独立路径进行最大路径比较的路径
    maxpaths.append([ (box[4],[ind]) for ind,box in enumerate(dets[-1])])
    for link_ind,link in enumerate(links[::-1]): #每一帧与后一帧的link，为一个list
        curmaxpaths=[]
        linkflags=np.zeros(len(maxpaths[0]),int)
        det_ind=len(links)-link_ind-1
        for ind,linkboxes in enumerate(link): #每一帧中每个box的link，为一个list
            if linkboxes == []:
                curmaxpaths.append((dets[det_ind][ind][4],[ind]))
                continue
            linkflags[linkboxes]=1
            prev_ind=np.argmax([maxpaths[0][linkbox][0] for linkbox in linkboxes])
            prev_score=maxpaths[0][linkboxes[prev_ind]][0]
            prev_path=copy.copy(maxpaths[0][linkboxes[prev_ind]][1])
            prev_path.insert(0,ind)
            curmaxpaths.append((dets[det_ind][ind][4]+prev_score,prev_path))
        root=[maxpaths[0][ind] for ind,flag in enumerate(linkflags) if flag == 0]
        roots.insert(0,root)
        maxpaths.insert(0,curmaxpaths)
    roots.insert(0,maxpaths[0])
    maxscore=0
    maxpath=[]
    for index,paths in enumerate(roots):
        if paths==[]:
            continue
        maxindex=np.argmax([path[0] for path in paths])
        if paths[maxindex][0]>maxscore:
            maxscore=paths[maxindex][0]
            maxpath=paths[maxindex][1]
            rootindex=index
    return rootindex,maxpath,maxscore

def rescore(dets, rootindex, maxpath, maxsum):
    newscore=maxsum/len(maxpath)
    for i,box_ind in enumerate(maxpath):
        dets[rootindex+i][box_ind][4]=newscore

def deleteLink(dets,links, rootindex, maxpath,thesh):
    for i,box_ind in enumerate(maxpath):
        areas=[(box[2]-box[0]+1)*(box[3]-box[1]+1) for box in dets[rootindex+i]]
        area1=areas[box_ind]
        box1=dets[rootindex+i][box_ind]
        x1=np.maximum(box1[0],dets[rootindex+i][:,0])
        y1=np.maximum(box1[1],dets[rootindex+i][:,1])
        x2=np.minimum(box1[2],dets[rootindex+i][:,2])
        y2=np.minimum(box1[3],dets[rootindex+i][:,3])
        w =np.maximum(0.0, x2 - x1 + 1)
        h =np.maximum(0.0, y2 - y1 + 1)
        inter = w * h
        ovrs = inter / (area1 + areas - inter)
        deletes=[ovr_ind for ovr_ind,ovr in enumerate(ovrs) if ovr >= IOU_THRESH] #保存待删除的box的index
        if rootindex+i<len(links): #除了最后一帧，置box_ind的box的link为空
            for delete_ind in deletes:
                links[rootindex+i][delete_ind]=[]
        if i > 0 or rootindex>0:
            for priorbox in links[rootindex+i-1]: #将前一帧指向box_ind的link删除
                for delete_ind in deletes:
                    if delete_ind in priorbox:
                        priorbox.remove(delete_ind)

def saveforAP(dets_all,frame_names):
    #保存结果为用于计算ap的格式
    for cls_ind, cls_name in enumerate(CLASSES):
        if cls_name == '__background__':
            continue
        dirpath='/workspace/liruiguang/imagenet/seqnms-results/'
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        filename = '{:s}{:s}.txt'.format(dirpath,cls_name)
        with open(filename, 'a') as f:
            for frame_ind, frame_name in enumerate(frame_names):
                dets = dets_all[cls_ind-1][frame_ind]
                for k in xrange(dets.shape[0]):
                    f.write('{:s} {:.6f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(frame_name, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))

def dsnms(video):
    dets=createInputs(video)
    links=createLinks(dets)
    maxPath(dets,links)
    NMS(dets)
    frame_names=sorted(video.keys())
    saveforAP(dets,frame_names)

pkllistfile=open('/workspace/liruiguang/imagenet/pkllist.txt')
pkllist=pkllistfile.readlines()
pkllistfile.close()
pkllist=[pkl.strip() for pkl in pkllist]
for pkl in pkllist:
    f = open('/workspace/liruiguang/imagenet/'+pkl)
    video = pickle.load(f)
    dsnms(video['dets'])

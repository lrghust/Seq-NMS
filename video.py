import _init_paths
import tensorflow as tf
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
from networks.factory import get_network


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


#CLASSES = ('__background__','person','bike','motorbike','car','bus')

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        cv2.rectangle(im,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255),2)
        cv2.putText(im,'{:s} {:.3f}'.format(class_name, score),(int(bbox[0]),int(bbox[1])+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.CV_AA)
        '''
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    '''


def demo(sess, net, video_name):
    v_file = os.path.join(cfg.DATA_DIR, 'demo', video_name)
    vc=cv2.VideoCapture()
    vc.open(v_file)
    ret,im=vc.read()
    fps=vc.get(5)
    height = np.size(im, 0)
    width = np.size(im, 1)
    vc.release()
    vc.open(v_file)
    allscores=[]
    allboxes=[]
    '''
    while(vc.isOpened()):
        ret,im = vc.read()
        if ret is False:
            break
        scores, boxes = im_detect(sess, net, im)
        allscores.append(scores)
        allboxes.append(boxes)
        ###
        print 'finish a frame'
        ###
    '''
    for i in range(6):
        ret,im=vc.read()
        scores, boxes = im_detect(sess, net, im)
        allscores.append(scores)
        allboxes.append(boxes)
        print 'finish a frame'

    vc.release()

    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    LINK_THRESH = 0.8
    alldets=[]
    for cls_ind, cls, in enumerate(CLASSES[1:]):
        cls_ind += 1
        dets=[]
        links=[]
        for index in range(1, len(allboxes)):
            box1 = allboxes[index - 1]
            box2 = allboxes[index]
            score1 = allscores[index - 1]
            score2 = allscores[index]
            cls_boxes = box1[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = score1[:, cls_ind]
            dets1 = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
            if index == 1:
                dets.append(dets1)
            cls_boxes = box2[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = score2[:, cls_ind]
            dets2 = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            dets.append(dets2)
            link={}
            for ind1,box1 in enumerate(dets1):
                link[ind1]=[]
                for ind2,box2 in enumerate(dets2):
                    x1=max(box1[0],box2[0])
                    y1=max(box1[1],box2[1])
                    x2=min(box1[2],box2[2])
                    y2=min(box1[3],box2[3])
                    area1=(box1[2]-box1[0]+1)*(box1[3]-box1[1]+1)
                    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
                    w = max(0.0, x2 - x1 + 1)
                    h = max(0.0, y2 - y1 + 1)
                    inter = w * h
                    ovr = inter / (area1 + area2 - inter)
                    #print ovr
                    if ovr >= LINK_THRESH:
                        link[ind1].append(ind2)
            links.append(link)
        #start from here-findmaxpath,rescore,remove
        #print links
        #exit()
        links.append({})
        while True:
            rootindex,maxpath,maxsum=findMaxPath(links,dets)
            if len(maxpath) <= 1:
                break
            rescore(dets,rootindex,maxpath,maxsum)
            deleteLink(links,rootindex,maxpath)
        alldets.append(dets)
    print 'finish rescoring'
    vc.open(v_file)
    vw=cv2.VideoWriter()
    vw.open('result.mp4',cv2.cv.FOURCC(*'MJPG'),fps,(width,height))
    for i in range(6):
        ret,im=vc.read()
        #im = im[:, :, (2, 1, 0)]
        #fig, ax = plt.subplots(figsize=(12, 12))
        #ax.imshow(im, aspect='equal')
        for ind,dets in enumerate(alldets):
            cls=CLASSES[ind+1]
            onedets=dets[i]
            keep = nms(onedets, NMS_THRESH)
            onedets = onedets[keep, :]
            vis_detections(im, cls, onedets, thresh=CONF_THRESH)
        vw.write(im)
    vw.release()

#link:dict links:list
def findMaxPath(links,dets):
    #start from here
    delete=[]
    maxpath=[]
    maxsum=0
    rootindex=0
    for link_ind,link in enumerate(links):
        roots = link.keys()
        delete=set(delete)
        for elem in delete:
            if elem in roots:
                roots.remove(elem)
        delete=[]
        for root in roots:
            delete.extend(link[root])
            curpath,cursum=findMaxPathFromOneRoot(links,dets,link_ind,root)
            if cursum > maxsum:
                maxpath=curpath
                maxsum=cursum
                rootindex=link_ind
    return rootindex,maxpath,maxsum

def findMaxPathFromOneRoot(links,dets,index,root):
    if root not in links[index] or links[index][root]==[]:
        return [root],dets[index][root][4]
    maxsum=0
    for nextroot in links[index][root]:
        curpath,cursum=findMaxPathFromOneRoot(links,dets,index+1,nextroot)
        if cursum > maxsum:
            maxsum=cursum
            maxpath=curpath
    maxsum+=dets[index][root][4]
    maxpath.insert(0,root)
    return maxpath,maxsum

def rescore(dets, rootindex, maxpath, maxsum):
    newscore=maxsum/len(maxpath)
    for i,box in enumerate(maxpath):
        onedets=dets[rootindex+i]
        onedets[box][4]=newscore

def deleteLink(links, rootindex, maxpath):
    for i,box in enumerate(maxpath):
        link=links[rootindex+i]
        link[box]=[]
        if i > 0:
            prior=links[rootindex+i-1]
            for key in prior.keys():
                if box in prior[key]:
                    prior[key].remove(box)


def getScore(frame_index, box_index, dets):
    return dets[frame_index][box_index][4]

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        default=' ')

    args = parser.parse_args()

    return args
if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    if args.model == ' ':
        raise IOError(('Error: Model not found.\n'))

    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    net = get_network(args.demo_net)
    # load model
    saver = tf.train.Saver()
    saver.restore(sess, args.model)
    #sess.run(tf.initialize_all_variables())

    print '\n\nLoaded network {:s}'.format(args.model)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(sess, net, im)

    video='test.mp4'
    demo(sess,net,video)

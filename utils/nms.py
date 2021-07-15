import numpy as np


def nms(dets, thres):
    '''

    :param dets: dets为一个k*5的数组，元素有4个坐标信息和一个得分
    :param thres: 阈值
    :return: 保留下来的结果

    '''

    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]

    areas = (y2-y1+1) * (x2-x1 + 1)
    scores = dets[:,4]
    keep = []

    order = scores.argsort()[::-1]

    while(order.size > 0):
        print('order:',order)
        i = order[0]
        keep.append(i)
        x11 = np.maximum(x1[i],x1[order[1:]])
        x22 = np.minimum(x2[i],x2[order[1:]])
        y11 = np.maximum(y1[i],y1[order[1:]])
        y22 = np.minimum(y2[i],y2[order[1:]])

        h = np.maximum(0,y22-y11+1)
        w = np.maximum(0,x22-x11+1)

        overlap = h * w

        ious = overlap / (areas[i] + areas[order[1:]] - overlap)
        print(ious)
        #得到小于thres阈值的idx
        idx = np.where(ious <= thres)[0]

        print(idx)
        order = order[idx+1]

    return keep

if __name__ =='__main__':
    boxes = np.array([[100, 100, 210, 210, 0.72],
                      [250, 250, 420, 420, 0.8],
                      [220, 220, 320, 330, 0.92],
                      [100, 100, 210, 210, 0.72],
                      [230, 240, 325, 330, 0.81],
                      [220, 230, 315, 340, 0.9]])
idx = nms(boxes,thres=0.7)
print(idx)
print(boxes[idx])
import numpy as np

def soft_nms(dets,thres_iou, sigma,thres_score, method):
    N = dets.shape[0]
    for i in range(dets.shape[0]):
        maxscore = dets[i,4]
        maxpos = i

        x1 = dets[i,0]
        y1 = dets[i,1]
        x2 = dets[i,2]
        y2 = dets[i,3]
        ts = dets[i,4]

        pos = i+1
        while pos < N:
            if dets[pos,4] > maxscore:
                maxscore = dets[pos,4]
                maxpos = pos
            pos += 1

        dets[i,0] = dets[maxpos,0]
        dets[i,1] = dets[maxpos,1]
        dets[i,2] = dets[maxpos,2]
        dets[i,3] = dets[maxpos,3]
        dets[i,4] = maxscore

        dets[maxpos,0] = x1
        dets[maxpos,1] = y1
        dets[maxpos,2] = x2
        dets[maxpos,3] = y2
        dets[maxpos,4] = ts

        x1 = dets[i,0]
        y1 = dets[i,1]
        x2 = dets[i,2]
        y2 = dets[i,3]
        ts = dets[i,4]
        area1 = (x2- x1 + 1)*(y2-y1+1)
        pos = i+1
        while pos < N:
            xx1 = dets[pos,0]
            yy1 = dets[pos,1]
            xx2 = dets[pos,2]
            yy2 = dets[pos,3]
            area2 = (xx2-xx1+1)*(yy2-yy1+1)
            w = max(0,min(xx2,x2) - max(xx1,x1)+1)
            h = max(0,min(yy2,y2) - max(yy1,y1)+1)

            overlap = w*h
            iou = overlap / (area1+area2-overlap)
            if(iou > thres_iou):
                if method == 1:
                    dets[pos,4] *= (1-iou)
                elif method == 2:
                    dets[pos,4] *= np.exp(-(iou*iou)/sigma)
                else:
                    dets[pos,4] = 0
            if dets[pos,4] < thres_score:
                dets[pos,4] = dets[N-1,4]
                dets[pos,0] = dets[N-1,0]
                dets[pos,1] = dets[N-1,1]
                dets[pos,2] = dets[N-1,2]
                dets[pos,3] = dets[N-1,3]
                N = N-1
                pos = pos -1
            pos = pos + 1


    keep = [i for i in range(N)]
    return keep


if __name__ == '__main__':
    boxes = np.array([[100, 100, 210, 210, 0.72],
                      [250, 250, 420, 420, 0.8],
                      [220, 220, 320, 330, 0.92],
                      [100, 100, 210, 210, 0.72],
                      [230, 240, 325, 330, 0.81],
                      [220, 230, 315, 340, 0.9]])

    print(soft_nms(boxes,0.7,0.5,0.1,1))
    print(soft_nms(boxes,0.7,0.5,0.1,3))
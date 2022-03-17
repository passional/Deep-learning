import math
import numpy as np

def iou(box1, box2):
    # box format: xyxy

    area1 = (box1[3] - box1[1]) * (box1[2] - box1[0])
    area2 = (box2[3] - box2[1]) * (box2[2] - box2[0])
    inter_area = (min(box1[2], box2[2]) - max(box1[0], box2[0])) * \
                 (min(box1[3], box2[3]) - max(box1[1], box2[1]))
    return inter_area / area1 + area2 - inter_area

def spm(iou, mode='linear', sigma=0.3):
    # score penalty mechanism (soft-nms)

    if mode == 'linear':
        return 1 - iou
    elif mode == 'gaussian':
        return math.e ** (- (iou ** 2) / sigma)
    else:
        raise NotImplementedError

def NMS(lists, conf_thre, iou_thre, soft = True, soft_thre = 0.001):
	#filter是过滤可迭代对象的东西，
	#先过滤一遍，比如2000个过滤到200个
	lists = filter(lambda x: x[4] >= conf_thre, lists)

	#排序
	lists = sorted(lists, key = lambda x: x[4], reverse = True)
	keep = []

	while lists:
		m = lists.pop(0)
		keep.append(m)
		#循环剩下的，依次和第一个做iou的计算，如果相似大于0.7，则剔除出去，否则留在这里，最后剩下的就是要的东西
		for i, pred in enumerate(lists):
			_iou = iou(m, pred)
			if _iou >= iou_thre:
				if soft:
					pred[4] *= spm(_iou, mode = 'gaussian',sigma = 0.3)
					keep.append(lists.pop[i])
				else:
					lists.pop(i)
	if soft:
        keep = list(filter(lambda x: x[4] >= soft_thre, keep))
        keep = sorted(keep, key=lambda x: x[4], reverse=True)

    return keep





if __name__ == '__main__':
	np.random.seed(0)
	#生成300行，2列的随机数组来代表
	x1y1 = np.random.randint(0, 300, (300, 2)) / 600
	x2y2 = np.random.randint(300, 600, (300, 2)) / 600
	#以列为轴拼接一下数组，组成300行4列的数组
	boxes = np.concatenate((x1y1, x2y2), 1)
	scores = np.random.rand(300, 1)
	scores = list(np.concatenate((boxes, scores), 1))
	detections = NMS(list, )
	print(len(detections), detections)
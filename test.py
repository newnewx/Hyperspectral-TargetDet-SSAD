from __future__ import print_function
import os
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import glob
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from metrics import StreamSegMetrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from loss.metrics import ROCMetric
import warnings

warnings.filterwarnings('ignore')

class Transformer(object):
    def __init__(self, size=None, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img_):
        # img_ = img_.resize(self.size, self.interpolation)
        img_ = self.toTensor(img_)  
        img_.sub_(0.5).div_(0.5)   
        return img_

def save_roc_curve(fpr,tpr,roc_auc,index): 

    plt.figure()
    lw = 4
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label=' AUC = %0.6f' % roc_auc) 
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right",fontsize=30)
    plt.savefig('roccc_%d.jpg' % index)
    # plt.savefig('fig.jpg')
    plt.show()

def false_alarm_rate(target, predicted):
    """
    calculate AUC and false alarm auc
    
    Input:
        target.shape = [n,1]
        predicted.shape = [n,1]
    
    Output:
        PD_PF_auc
        PF_tau_auc
    """
    
    predicted = (predicted - predicted.min())/(1e-12 + predicted.max() - predicted.min())
    PF, PD, taus = roc_curve(target, predicted)
    PD_PF_auc = np.trapz(PD.squeeze(),PF.squeeze())
    #area0 = np.trapz(PD.squeeze(),PF.squeeze())
    PD_tau_auc = - np.trapz(PD.squeeze(),taus.squeeze())
    PF_tau_auc = - np.trapz(PF.squeeze(),taus.squeeze())
    return PD_PF_auc, PD_tau_auc, PF_tau_auc


def main(model, model_path, save_dir, model_name):
	print(model_name)
	# model.load_state_dict(torch.load(model_path,map_location='cpu'))
	model.load_state_dict(torch.load(model_path,map_location='cuda:3'),False)
	test_img = sorted(glob.glob('./hyper-images/*.png')) #img
	test_lab = sorted(glob.glob('./hyper-images/*.tif')) #groundtruth

	cnt = 0
	iou_tol = 0
	roc_x=np.zeros(51)
	roc_y=np.zeros(51)

	metrics = StreamSegMetrics(2)
	metrics.reset()
	
	for j in range(0,len(test_img)):
		x_batch = test_img[j]
		folder_path, file_name = os.path.split(x_batch)

		test_image = Image.open(x_batch).convert('RGB')
		h=test_image.size[0]

		test_image = test_image.resize((256,256),Image.BILINEAR)

		transformer = Transformer((256, 256))
		# transformer = Transformer()
		img = transformer(test_image)
		img = img.unsqueeze(0)
		img = Variable(img)

		y_batch = test_lab[j]
		test_label = Image.open(y_batch) #.convert('RGB')

		test_label = test_label.resize((h,h), Image.BILINEAR) 

		label = torch.from_numpy(np.array(test_label)/255.0)
		
		pred_image= model(img)    #tensor
		# pred_image, output_sr= model(img)
		pred_image = pred_image.squeeze(0) 

		### metric ###
		label = label.cpu().numpy()
		pred = pred_image[1].detach().cpu().numpy()
		pred = pred.astype(np.float64)    #array  float
		label = label.astype(np.int64)

		to_tensor = transforms.ToTensor()  
		pred = Image.fromarray(pred)   #PIL
		pred =pred.resize((h,h), Image.BILINEAR)
		pred= to_tensor(pred)
		pred=pred.cpu().numpy()
		pred_ = pred.reshape(-1)
		label_ = label.reshape(-1)
		a,b,c=false_alarm_rate(label_,pred_)

		print(file_name + '  AUC=%.4f, PD=%.4f, PF=%.4f' % (a,b,c))

		pred = Image.fromarray(pred[0]*255)

		pred.convert('L').save( save_dir + file_name[:-4]  + model_name + '.png' )


	iou_final = iou_tol/len(test_img)
	print("final iou = ", iou_final)
	print()
	
"""
import models.res18_atten as res18_atten
import models.seg_net as seg_net
import models.u_net as u_net
import models.res_net as res_net
import models.seg_bottle as seg_bottle
"""
# import models.segnet as network
#import models.seg_bottle_half as network
import models.model as model
#model = res18_atten.Scattnet(3, 3)
#model_path = './checkpoint/res18_atten/netS_e18_acc0.97.pth'
#save_dir = './hyper-images/result/res18_atten/'

if __name__ == '__main__':

	main(model.network(3,2), './checkpoint/200/model_best_80.pth', './result/', '_200')


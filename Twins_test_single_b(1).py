from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
print(torch.__version__)
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
#from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from matplotlib import pyplot
from scipy.signal import savgol_filter
from facenet_pytorch import InceptionResnetV1
from facenet_pytorch.models.inception_resnet_v1 import BasicConv2d




class MyImgFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        return super(MyImgFolder, self).__getitem__(index), self.imgs[index]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("IM HERE")


trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization,
    #transforms.Resize((160,160))
])



def predictions(dataloader):
    score = []
    y = []
    file=[]
    for idx,(data, filename) in enumerate(dataloader):
        x=data[0]
        label=data[1]
        x = x.to(device)
        label = label.to(device)
        output = resnet(x)
        y_pred = torch.nn.functional.softmax(output,dim=1)

        for z in label:
            y.append(z.cpu().data.item())

        for f in filename[0]:
            file.append(f)

        for idx, item in enumerate(y_pred):
            score.append(item[0].item())


    return score,y


################################## plot ###
def get_fpr_tpr(score1, y1):
    # false positive rate
    fpr1 = []
    # true positive rate
    tpr1 = []
    fnr1 = []
    # Iterate thresholds from 0.0, 0.01, ... 1.0
    thresholds = np.arange(-0.1, 1.01, .00001)
    #thresholds = np.linspace(np.amin(score1), np.amax(score1), num=10000)

    # get number of positive and negative examples in the dataset
    N1 = sum(y1)
    P1 = len(y1) - N1

    # iterate through all thresholds and determine fraction of true positives
    # and false positives found at this threshold

    for thresh in thresholds:
        FP1 = 0
        TP1 = 0
        for i in range(len(score1)):
            if (score1[i] > thresh):
                if y1[i] == 0:
                    TP1 = TP1 + 1
                if y1[i] == 1:
                    FP1 = FP1 + 1
        FN1 = P1 - TP1
        fnr1.append(FN1 / float(P1))
        fpr1.append(FP1 / float(N1))
        tpr1.append(TP1 / float(P1))

    fpr1 = np.array(fpr1)
    fnr1 = np.array(fnr1)
    tpr1 = np.array(tpr1)

    fpr1 = fpr1 * 100.0
    fnr1 = fnr1 * 100.0
    tpr1 = tpr1 * 100.0
    EER = []
    BPCER10 = []
    BPCER30 = []
    BPCER5 = []
    APCER10 = []
    APCER5 = []
    APCER30 = []
    for idx, item in enumerate(fnr1):

        if round(fnr1[idx]) == round(fpr1[idx]) and len(EER) != 1:
            EER.append(fnr1[idx])

        if round(fpr1[idx]) == 10 and len(BPCER10) != 1:
            BPCER10.append(fnr1[idx])
        if round(fpr1[idx]) == 5 and len(BPCER5) != 1:
            BPCER5.append(fnr1[idx])
        if round(fpr1[idx]) == 1 and len(BPCER30) != 1:
            BPCER30.append(fnr1[idx])

        if round(fnr1[idx]) == 10 and len(APCER10) != 1:
            APCER10.append(fpr1[idx])
        if round(fnr1[idx]) == 5 and len(APCER5) != 1:
            APCER5.append(fpr1[idx])
        if round(fnr1[idx]) == 1 and len(APCER30) != 1:
            APCER30.append(fpr1[idx])
    EER = np.array(EER)
    BPCER10 = np.array(BPCER10)
    BPCER20 = np.array(BPCER30)

    print("EER: ")
    print(EER)
    print("BPCER10: ")
    print(BPCER10)
    print("BPCER5: ")
    print(BPCER5)
    print("BPCER 1: ")
    print(BPCER30)

    print("APCER10: ")
    print(APCER10)
    print("APCER 5: ")
    print(APCER5)
    print("APCER 1: ")
    print(APCER30)
    return fpr1, fnr1, tpr1


stnd_facemorpher = "/home/kelsey/Desktop/TRAINING_TESTING/DRD/pert/"
stnd_opencv = "/home/kelsey/Desktop/TRAINING_TESTING/DRD/Original/"


CURRENTMORPH = 'TWINS BASELINE'
class_sample_count = [5701,3221]
#class_sample_count= [500,1499]
#class_sample_count = [183,131]
batch_size=1

stnd_facemorph = MyImgFolder(stnd_facemorpher,transform=trans)
stnd_cv= MyImgFolder(stnd_opencv,transform=trans)

def get_sampler(class_sample_count,dataset):
    weights = 1.0 / torch.tensor(class_sample_count, dtype=torch.float)
    target_list = torch.tensor(dataset.targets)
    target_list = target_list[torch.randperm(len(target_list))]
    weights = weights[target_list]
    print("WEIGHTS " + str(len(weights)))
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,len(weights))
    return sampler

st_fm = torch.utils.data.DataLoader(stnd_facemorph, batch_size=batch_size, sampler=get_sampler([747,374],stnd_facemorph))
st_cv= torch.utils.data.DataLoader(stnd_cv, batch_size=batch_size)#, sampler=sampler2)


def test(epoch):
    score,y = predictions(trainloader)

    print("LOADED PREDICTIONS>")
    print("GETTING FPR / TPR RATES>>>")
    fpr, fnr, tpr, = get_fpr_tpr(score, y)
    print("CALCULATING AUC>>>")
    auc = -1 * np.trapz(tpr / 100.0, fpr / 100.0)
    print(auc)
    #print(fpr/100, tpr/100, auc)
    return fpr/100,tpr/100, auc

resnet = InceptionResnetV1(
    classify=True,
    pretrained='vggface2',
    num_classes=2
)

#resnet.conv2d_1a = BasicConv2d(1, 32, kernel_size=3, stride=2)
resnet=resnet.to(device)
resnet = torch.nn.DataParallel(resnet)
cudnn.benchmark = True


resnet.load_state_dict(torch.load('TWINS_LANDMARK_512_RGB_BALANCED_BATCH32_LR1e-06_MARGIN1.0_22_WEIGHTS_VALID_TWINS_CPPMTCNN_FINETUNE.pth',map_location=device))
resnet.eval()


#### RUN MODEL ####
EPOCHS = 1


trainloader = st_cv
cv_feret_fpr, cv_feret_tpr,CV_feret_AUC= test(EPOCHS)
print("pert")
del trainloader

trainloader = st_fm
feret_FM_fpr, feret_FM_tpr,FM_feret_AUC = test(EPOCHS)
print("FACEMORPHER")
del trainloader

plt.plot(cv_feret_fpr,cv_feret_tpr,color='b',label='FRGC AUC=%0.6f'% CV_feret_AUC)
plt.plot(feret_FM_fpr,feret_FM_tpr,color='g', label='PERTURBED FRGC AUC=%0.6f' % FM_feret_AUC)




plt.legend(loc='upper right')
plt.plot([0,1],[0,1],'r--')
plt.ylim([0,1.01])
plt.title("DRD DATASET")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("DRD_PERT")

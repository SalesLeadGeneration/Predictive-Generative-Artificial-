import cv2 as cv
from numpy import matlib
from AOA import AOA
from DOA import DOA
from Glob_Vars import Glob_Vars
from Image_Results import Image_Results
from Model_ADLabV3p import Model_ADLabV3p
from Model_Attention_Guide import Model_Attention_guidedDNN
from Model_CNN import Model_CNN
from Model_GLDD_GADAN import Model_GLDD_GADAN
from Model_ICNN import Model_ICNN
from Model_ResiUNet import Model_ResiUNet
from Model_Resnet import Model_Resnet
from Model_UNet import Model_UNet
from Model_VGG19 import Model_VGG19
from NGO import NGO
from Objection_Function import Objfun_seg
from PROPOSED import PROPOSED
from Plot_Results import *
from Plot_Segmentation import *
from SHVO import SHVO

# Read Dataset
an = 0
if an == 1:
    path_img = './Dataset/Images'
    path_mask = './Dataset/Annotations'
    img_dir = os.listdir(path_img)
    Mask_dir = os.listdir(path_mask)
    Images = []
    GT = []
    for k in range(len(img_dir)):
        print(k)
        Image_path = path_img + '/' + img_dir[k]
        Mask_path = path_mask + '/' + Mask_dir[k]
        if Mask_dir[k] == img_dir[k]:
            img = cv.imread(Image_path)
            img = cv.resize(img, [512, 512])
            gt = cv.imread(Mask_path)
            gt = cv.resize(gt, [512, 512])
            Images.append(img)
            GT.append(gt)
    np.save('Images.npy', Images)
    np.save('Groundtruth.npy', GT)

# Generate Target
an = 0
if an == 1:
    Tar = []
    Ground_Truth = np.load('Groundtruth.npy', allow_pickle=True)
    for i in range(len(Ground_Truth)):
        print(i)
        image = Ground_Truth[i]
        if np.count_nonzero(image == 255) <= 0:
            Tar.append(0)  # Normal
        elif np.count_nonzero(image == 255) <= 3000:
            Tar.append(1)  # Early Stage
        elif (np.count_nonzero(image == 255) > 3000) & (np.count_nonzero(image == 255) <= 7000):
            Tar.append(2)  # Early Severe Stage
        elif (np.count_nonzero(image == 255) > 7000) & (np.count_nonzero(image == 255) <= 13000):
            Tar.append(3)  # Mid Stage
        else:
            Tar.append(4)  # Severe Stage
    # unique code
    Target = np.asarray(Tar)
    uniq, Counts = np.unique(Target, return_counts=True)
    target = np.zeros((Target.shape[0], len(uniq)))  # create within rage zero values
    for uni in range(len(uniq)):
        index = np.where(Target == uniq[uni])
        target[index[0], uni] = 1
    np.save('Target.npy', target)

# Optimization for Segmentation
an = 0
if an == 1:
    Images = np.load('Images.npy', allow_pickle=True)
    GT = np.load('Groundtruth.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Glob_Vars.Images = Images
    Glob_Vars.GT = GT
    Glob_Vars.Target = Target
    Npop = 10
    Chlen = 3  # Hidden Neuron Count, Learning rate, Steps per epoch
    xmin = matlib.repmat([5, 0.01, 100], Npop, 1)
    xmax = matlib.repmat([255, 0.99, 500], Npop, 1)
    fname = Objfun_seg
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("NGO...")
    [bestfit1, fitness1, bestsol1, time] = NGO(initsol, fname, xmin, xmax, Max_iter)  # NGO

    print("DOA...")
    [bestfit2, fitness2, bestsol2, time1] = DOA(initsol, fname, xmin, xmax, Max_iter)  # DOA

    print("AOA...")
    [bestfit3, fitness3, bestsol3, time2] = AOA(initsol, fname, xmin, xmax, Max_iter)  # AOA

    print("SHVO...")
    [bestfit4, fitness4, bestsol4, time3] = SHVO(initsol, fname, xmin, xmax, Max_iter)  # SHVO

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time4] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # PROPOSED

    BestSol = ([bestsol1, bestsol2, bestsol3, bestsol4, bestsol5])
    fitness = ([fitness1.ravel(), fitness2.ravel(), fitness3.ravel(), fitness4.ravel(), fitness5.ravel()])

    np.save('BestSol.npy', BestSol)
    np.save('Fitness.npy', fitness)

# Segmentation
an = 0
if an == 1:
    Images = np.load('Images.npy', allow_pickle=True)
    GT = np.load('Groundtruth.npy', allow_pickle=True)
    BestSol = np.load('BestSol.npy', allow_pickle=True)[4, :]
    EVAL = []
    Epoch = [30, 60, 90, 120, 150]
    for learn in range(len(Epoch)):
        Eval = np.zeros((10, 15))
        for j in range(5):  # For Algorithms
            print(learn, j)
            Eval[j, :], Image = Model_ADLabV3p(Images, GT, BestSol.astype('int'))
        Eval[5, :] = Model_UNet(Images, GT)
        Eval[6, :] = Model_ResiUNet(Images, GT)
        Eval[7, :] = Model_ICNN(Images, GT)
        Eval[8, :] = Model_ADLabV3p(Images, GT)
        Eval[9, :] = EVAL[4, :]
        EVAL.append(Eval)
        np.save('Eval_all_seg.npy', EVAL)  # Save the Eval_all
        np.save('Seg_Images.npy', Image)

# Classification
an = 0
if an == 1:
    k_fold = 5
    Images = np.load('Seg_Images.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Eval = []
    for i in range(k_fold):
        Total_Index = np.arange(Images.shape[0])
        Test_index = np.arange(((i - 1) * (Images.shape[0] / k_fold)) + 1, i * (Images.shape[0] / k_fold))
        Train_Index = np.setdiff1d(Total_Index, Test_index)
        train_data = Images[Train_Index, :]
        train_target = Target[Train_Index, :]
        test_data = Images[Test_index.astype('int'), :]
        test_target = Target[Test_index.astype('int'), :]
        EVAL = np.zeros((5, 25))
        EVAL[0, :] = Model_Attention_guidedDNN(train_data, train_target, test_data, test_target, i)
        EVAL[1, :] = Model_CNN(train_data, train_target, test_data, test_target, i)
        EVAL[2, :] = Model_Resnet(train_data, train_target, test_data, test_target, i)
        EVAL[3, :] = Model_VGG19(train_data, train_target, test_data, test_target, i)
        Eval[4, :] = Model_GLDD_GADAN(train_data, train_target, test_data, test_target, i)
        Eval.append(EVAL)
    np.save('Eval_KFOLD.npy', Eval)

# Image_Results()
ROC_curve()
Plot_table()
Plot_Results()
Kfold_proposed()
plot_results_conv()
Plot_Results_seg_method()
plot_results_seg_table()

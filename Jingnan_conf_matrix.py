import numpy as np
import warnings
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
warnings.simplefilter('ignore')
import cv2
cv2.setNumThreads(0)


path = "E:\\AIMIRA_final_treatment_effect_code\\create_mask_image_new\\mask_bones\\pred_original_image\\"
organ_ls = [
    # 'bones',
    #        'skin',
    #        'vessel',
    #        'tendons',
    #        'othertissue',
            'TSY',
            'SYN',
            ]

for organ in organ_ls:
    TP_TP = 0
    TP_FN = 0
    FP_FP = 0
    FP_TN = 0
    TN_TN = 0
    TN_FP = 0
    FN_FN = 0
    FN_TP = 0
    for i in range (1,11):
        print(f"---------organ: {organ}, repeat: {i}--------")
        groundtruth =   np.load('Jingnan/Jingnan_repeat_3_label_fold'+str(i)+'_groundtruth.npy')
        original_pred = np.load(path + 'repeat_3_predict_label_fold'+str(i)+'_orginal.npy')
        bone_blocked_pred = np.load(f'Jingnan/Jingnan_repeat_3_predict_label_fold{i}_exclude_{organ}.npy')
        # groundtruth =   np.load('TH_repeat_3_label_fold'+str(i)+'_groundtruth.npy')
        # original_pred = np.load(path + 'repeat_3_predict_label_fold'+str(i)+'_orginal.npy')
        # bone_blocked_pred = np.load(f'TH_repeat_3_predict_label_fold{i}_bone.npy')
        # print(groundtruth)
        # print(original_pred)
        # print(bone_blocked_pred)

        # if i in [2, 3, 10]:
        #     if i == 2:
        #         idx = 11
        #     elif i == 3:
        #         idx = 6
        #     else:
        #         idx = 12
        #     groundtruth = np.array([value for i,value in enumerate(groundtruth) if i !=idx])
        #     original_pred = np.array([value for i,value in enumerate(original_pred) if i !=idx])
        #     bone_blocked_pred = np.array([value for i,value in enumerate(bone_blocked_pred) if i !=idx])

        for j in range (len(original_pred)):

            if groundtruth[j] == 1 and original_pred[j] == 1 and bone_blocked_pred[j] == 1:
              TP_TP = TP_TP + 1
            if groundtruth[j] == 1 and original_pred[j] == 1 and bone_blocked_pred[j] == 0:
              TP_FN = TP_FN + 1
            if groundtruth[j] == 0 and original_pred[j] == 1 and bone_blocked_pred[j] == 1:
              FP_FP = FP_FP + 1
              print('fold=', i)
              print('not helping patients FP_FP =', j)
            if groundtruth[j] == 0 and original_pred[j] == 1 and bone_blocked_pred[j] == 0:
              FP_TN = FP_TN + 1
              print('fold=', i)
              print('confusing patients FP_TN =', j)
            if groundtruth[j] == 0 and original_pred[j] == 0 and bone_blocked_pred[j] == 0:
              TN_TN = TN_TN + 1
            if groundtruth[j] == 0 and original_pred[j] == 0 and bone_blocked_pred[j] == 1:
              TN_FP = TN_FP + 1
            if groundtruth[j] == 1 and original_pred[j] == 0 and bone_blocked_pred[j] == 0:
              FN_FN = FN_FN + 1
              print('fold=', i)
              print('not helping patients FN_FN =', j)
            if groundtruth[j] == 1 and original_pred[j] == 0 and bone_blocked_pred[j] == 1:
              FN_TP = FN_TP + 1
              print('fold=', i)
              print('confusing patients FN_TP =', j)

    print('TP_TP = ', TP_TP)
    print('TP_FN = ', TP_FN)
    print('FP_FP = ', FP_FP)
    print('FP_TN = ', FP_TN)
    print('TN_TN = ', TN_TN)
    print('TN_FP = ', TN_FP)
    print('FN_FN = ', FN_FN)
    print('FN_TP = ', FN_TP)

    irrelevant = TP_TP + TN_TN
    essential = TP_FN + TN_FP
    confusing = FP_TN + FN_TP
    not_helping = FP_FP + FN_FN

    print('irrelevant=', irrelevant)
    print('essential=', essential )
    print('confusing=', confusing)
    print('not_helping=', not_helping)

    print('%irrelevant=', irrelevant*100/185)
    print('%essential=', essential*100/185 )
    print('%confusing=', confusing*100/185)
    print('%not_helping=', not_helping*100/185)


    # for i in range (64):
    #     tn, fp, fn, tp = confusion_matrix(all_original[:,i], all_predict[:,i]).ravel()
    #     print('confusion_'+str(i)+'_tn = %s_fp = %s_fn = %s_tp = %s_',tn, fp, fn, tp )
    # print(np.shape(all_original))
    # print(np.shape(all_predict))

import os
import sys
import numpy as np

from . import utils
import nibabel as nib

def volgen(vol_names, batch_size=1, return_segs=False, np_var='vol', pad_shape=None, resize_factor=1, add_feat_axis=True):
    """
    Base generator for random volume loading. Corresponding segmentations are additionally
    loaded if return_segs is set to True. If loading segmentations, it's expected that
    vol_names is a list of npz files with 'vol' and 'seg' arrays.

    Parameters:
        vol_names: List of volume files to load.
        batch_size: Batch size. Default is 1.
        return_segs: Loads corresponding segmentations. Default is False.
        np_var: Name of the volume variable if loading npz files. Default is 'vol'.
        pad_shape: Zero-pads loaded volumes to a given shape. Default is None.
        resize_factor: Volume resize factor. Default is 1.
        add_feat_axis: Load volume arrays with added feature axis. Default is True.
    """
    while True:
        # generate <batchsize> random image indices
        indices = np.random.randint(len(vol_names), size=batch_size)

        # load volumes and concatenate
        load_params = dict(np_var=np_var, add_batch_axis=True, add_feat_axis=add_feat_axis, pad_shape=pad_shape, resize_factor=resize_factor)
        imgs = [utils.load_volfile(vol_names[i], **load_params) for i in indices]
        vols = [np.concatenate(imgs, axis=0)]

        # optionally load segmentations and concatenate
        if return_segs:
            load_params['np_var'] = 'seg'  # be sure to load seg
            segs = [utils.load_volfile(vol_names[i], **load_params) for i in indices]
            vols.append(np.concatenate(segs, axis=0))

        yield tuple(vols)


def scan_to_scan(vol_names, bidir=False, batch_size=1, prob_same=0, no_warp=False, **kwargs):
    """
    Generator for scan-to-scan registration.

    Parameters:
        vol_names: List of volume files to load.
        bidir: Yield input image as output for bidirectional models_old. Default is False.
        batch_size: Batch size. Default is 1.
        prob_same: Induced probability that source and target inputs are the same. Default is 0.
        no_warp: Excludes null warp in output list if set to True (for affine training). Default if False.
        kwargs: Forwarded to the internal volgen generator.
    """
    zeros = None
    gen = volgen(vol_names, batch_size=batch_size, **kwargs)
    while True:
        scan1 = next(gen)[0]
        scan2 = next(gen)[0]

        # some induced chance of making source and target equal
        if prob_same > 0 and np.random.rand() < prob_same:
            if np.random.rand() > 0.5:
                scan1 = scan2
            else:
                scan2 = scan1

        # cache zeros
        if not no_warp and zeros is None:
            shape = scan1.shape[1:-1]
            zeros = np.zeros((batch_size, *shape, len(shape)))

        invols  = [scan1, scan2]
        outvols = [scan2, scan1] if bidir else [scan2]
        if not no_warp:
            outvols.append(zeros)

        yield (invols, outvols)


def  spineregnetgen(vol_names,labels=[6,5,4,3,2],batch_size=1):
    zeros = None
    def split_seg(seg):
        prob_seg = np.zeros((*seg.shape[:4], len(labels)))
        for i, label in enumerate(labels):
            prob_seg[0, ..., i] = seg[0, ..., 0] == label
        return prob_seg
    while True:
        idxes = np.random.randint(len(vol_names), size=batch_size)
        X_data = []
        Y_data = []
        L1_segdata = []
        L2_segdata = []
        L3_segdata = []
        L4_segdata = []
        L5_segdata = []
        YL1_segdata = []
        YL2_segdata = []
        YL3_segdata = []
        YL4_segdata = []
        YL5_segdata = []

        for idx in idxes:
                X = nib.load(vol_names[idx]).get_data()
                X = X[np.newaxis, ..., np.newaxis]
                X_data.append(X)
                Y = nib.load(vol_names[idx].replace("mr", "ct")).get_data()
                Y = Y[np.newaxis, ..., np.newaxis]
                Y_data.append(Y)

                X_seg=nib.load(vol_names[idx].replace(".nii", "_mask.nii")).get_data()
                X_seg = X_seg[np.newaxis, ..., np.newaxis]
                X_seg=split_seg(X_seg)
                tL1=X_seg[:, :, :, :, 0]
                tL1=tL1[..., np.newaxis]
                tL2=X_seg[:, :, :, :, 1]
                tL2=tL2[..., np.newaxis]
                tL3=X_seg[:, :, :, :, 2]
                tL3=tL3[..., np.newaxis]
                tL4=X_seg[:, :, :, :, 3]
                tL4=tL4[..., np.newaxis]
                tL5=X_seg[:, :, :, :, 4]
                tL5=tL5[..., np.newaxis]
                L1_segdata.append(tL1)
                L2_segdata.append(tL2)
                L3_segdata.append(tL3)
                L4_segdata.append(tL4)
                L5_segdata.append(tL5)

                Y_seg =nib.load(vol_names[idx].replace("mr", "ct").replace(".nii", "_mask.nii")).get_data()
                Y_seg = Y_seg[np.newaxis, ..., np.newaxis]
                Y_seg=split_seg(Y_seg)
                tYL1=Y_seg[:, :, :, :, 0]
                tYL1=tYL1[..., np.newaxis]
                tYL2=Y_seg[:, :, :, :, 1]
                tYL2=tYL2[..., np.newaxis]
                tYL3=Y_seg[:, :, :, :, 2]
                tYL3=tYL3[..., np.newaxis]
                tYL4=Y_seg[:, :, :, :, 3]
                tYL4=tYL4[..., np.newaxis]
                tYL5=Y_seg[:, :, :, :, 4]
                tYL5=tYL5[..., np.newaxis]

                YL1_segdata.append(tYL1)
                YL2_segdata.append(tYL2)
                YL3_segdata.append(tYL3)
                YL4_segdata.append(tYL4)
                YL5_segdata.append(tYL5)

        return_valsX = [np.concatenate(X_data, 0)]
        return_valsY = [np.concatenate(Y_data, 0)]
        return_L1_segdata=[np.concatenate(L1_segdata, 0)]
        return_L2_segdata=[np.concatenate(L2_segdata, 0)]
        return_L3_segdata=[np.concatenate(L3_segdata, 0)]
        return_L4_segdata=[np.concatenate(L4_segdata, 0)]
        return_L5_segdata=[np.concatenate(L5_segdata, 0)]
        return_YL1_segdata=[np.concatenate(YL1_segdata, 0)]
        return_YL2_segdata=[np.concatenate(YL2_segdata, 0)]
        return_YL3_segdata=[np.concatenate(YL3_segdata, 0)]
        return_YL4_segdata=[np.concatenate(YL4_segdata, 0)]
        return_YL5_segdata=[np.concatenate(YL5_segdata, 0)]

        if zeros is None:
            volshape = X.shape[1:-1]
            zeros = np.zeros((batch_size, *volshape, len(volshape)))

        yield ([return_valsX[0], return_valsY[0],
                return_L1_segdata[0],
                return_L2_segdata[0],
                return_L3_segdata[0],
                return_L4_segdata[0],
                return_L5_segdata[0]
                ,return_YL1_segdata[0],
                return_YL2_segdata[0],
                return_YL3_segdata[0],
                return_YL4_segdata[0],
                return_YL5_segdata[0]],
               [return_valsY[0], zeros,
                return_YL1_segdata[0],
                return_YL2_segdata[0],
                return_YL3_segdata[0],
                return_YL4_segdata[0],
                return_YL5_segdata[0]
                , return_YL1_segdata[0],
                return_YL2_segdata[0],
                return_YL3_segdata[0],
                return_YL4_segdata[0],
                return_YL5_segdata[0]
                ,return_YL1_segdata[0],
                return_YL2_segdata[0],
                return_YL3_segdata[0],
                return_YL4_segdata[0],
                return_YL5_segdata[0]])
                
                

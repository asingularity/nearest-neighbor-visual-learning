import numpy as np
import cv2


ALLOW_OVERLAP = True


def make_im(input_arrays, num_bins_per_pixel, input_im_dim, im_final_dim=1600, mod_for_disp=5, normalize_weights=False, borders_px=3):

    tmp_im_all = None
    tmp_im = None

    rf_ims_dict = {}

    #arr, weights = _collapse_binned_columns_to_pixels(arr_exp=input_arrays,
    #                                                  num_bins_per_pixel=num_bins_per_pixel)
    arr = input_arrays
    weights = input_arrays

    im_len = int(arr.shape[1] / 2)

    for r_tmp in range(weights.shape[0]):  # per rf

        arr_p = arr[r_tmp, 0:im_len]
        arr_n = arr[r_tmp, im_len::]

        #im0 = (0.5 - arr_n + arr_p).reshape((input_im_dim, input_im_dim))  # rf_dim
        #im1 = im0.copy()

        im0 = arr_p.reshape((input_im_dim, input_im_dim))  # rf_dim
        im1 = arr_n.reshape((input_im_dim, input_im_dim))  # rf_dim

        if normalize_weights:
            max_1 = np.amax(im1)
            max_0 = np.amax(im0)

            max_all = max(max_1, max_0)

            im1 = 0.5 + im0 * 0.5/max_all - im1 * 0.5/max_all

            im0 = im0 * 1.0 / max_0

        # im1 = 0.5 * (im1 + 1)

        spacer1 = 0.2 * np.ones((im0.shape[0], borders_px))

        use_both_ims = False  # i.e. if you want to also display only one of the lobes
        if use_both_ims:
            tmp2 = np.hstack((im0, spacer1, im1))
        else:
            tmp2 = im1

        rf_ims_dict[r_tmp] = tmp2.copy()

        if tmp_im is None:
            tmp_im = tmp2.copy()
        else:
            tmp3 = 0.2 * np.ones((borders_px, tmp_im.shape[1]))
            tmp_im = np.vstack((tmp_im, tmp3, tmp2))

        if (weights.shape[0] == 1) or (r_tmp > 0 and (r_tmp + 1) % mod_for_disp == 0):
            if tmp_im_all is None:
                tmp_im_all = tmp_im.copy()
            else:
                spacer = 0.8 * np.ones((tmp_im_all.shape[0], borders_px))
                tmp_im_all = np.hstack((tmp_im_all, spacer, tmp_im))

            tmp_im = None

    tmp_all_im = tmp_im_all

    max_dim = max(tmp_all_im.shape[0], tmp_all_im.shape[1])
    imscale = im_final_dim / max_dim  # 0.2: full table, 2.0
    tmp_im = cv2.resize(tmp_all_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

    return tmp_im, rf_ims_dict


import os
import scipy.io as sio
import numpy as np
import cv2
import time

def dude_Constrained3LevelHGS(visionPath, isPreCalculated, isWeightExist):
    # Videos in VISION folder have 3 parent folders
    videos = [os.path.join(root, file) for root, _, files in os.walk(visionPath) for file in files if file.endswith('p')]


    for i, videoPath in enumerate(videos):
        start_time = time.time()
        videoResPath = os.path.join(os.path.dirname(videoPath), os.path.basename(videoPath))
        print(f'Running video is {videoResPath}')

        if not isPreCalculated:
            cropped_prnu = None
            fp = sio.loadmat(os.path.join(videoResPath, 'fp.mat'))
            calculate_Transforms(videoPath, fp['cropped_prnu'], videoResPath)

        find_Result(videoPath, videoResPath, 2, 2, 42, isWeightExist)
        print(f'Elapsed time: {time.time() - start_time} seconds')

    result(visionPath)

def calculate_Transforms(videoPath, cropped_prnu, videoResPath):
    frames = [os.path.join(videoPath, file) for file in os.listdir(videoPath) if file.endswith('.mat') and file != 'db.mat']
    crop_size = 250
    ncc_range = 65
    y_size = cropped_prnu.shape[0] // 2
    x_size = cropped_prnu.shape[1] // 2

    transform = []
    transform_1 = []
    transform_2 = []

    for i, frame_path in enumerate(frames[:11]):
        Noisex = sio.loadmat(frame_path)['PRNU_image']
        print(i)

        Fp_camera_1 = cropped_prnu[(y_size-crop_size-ncc_range):(y_size+crop_size+ncc_range), (x_size-crop_size-ncc_range):(x_size+crop_size+ncc_range)]
        Noisex_ref = Noisex[(y_size-crop_size):(y_size+crop_size), (x_size-crop_size):(x_size+crop_size)]

        transform = try_transform(Fp_camera_1, Noisex_ref, 4, transform, i, ncc_range, np.zeros((1, 11)))
        print(f'Trying coarse sample points for frame {i} finished')

        temp_transforms4frame = [t for t in transform if t[0] == i]
        temp_transforms4frame.sort(key=lambda x: x[1])
        best_sample_transforms = temp_transforms4frame[-5:]

        for point_index in range(5):
            transform_center = best_sample_transforms[point_index]
            transform_1 = try_transform(Fp_camera_1, Noisex_ref, 2, transform_1, i, ncc_range, transform_center)
        print(f'Trying finer sample points for frame {i} finished')

        temp_transforms4frame = [t for t in transform_1 if t[0] == i]
        temp_transforms4frame.sort(key=lambda x: x[1])
        best_sample_transforms = temp_transforms4frame[-5:]

        for point_index in range(5):
            transform_center = best_sample_transforms[point_index]
            transform_2 = try_transform(Fp_camera_1, Noisex_ref, 1, transform_2, i, ncc_range, transform_center)

    sio.savemat(os.path.join(videoResPath, 'all_transforms.mat'), {'transform': transform, 'transform_1': transform_1, 'transform_2': transform_2})
    print('All transforms finished ...')

def try_transform(Fp_camera_1, Noisex_ref, step, transform, frame_id, ncc_range, transform_center):
    transform_count = len(transform)
    transform_center_point = transform_center[0, 3:11].reshape((4, 2))
    ncc_range2 = 2 * ncc_range

    for x1 in range(-1, 2):
        for x2 in range(-1, 2):
            for x3 in range(-1, 2):
                for y1 in range(-1, 2):
                    for y2 in range(-1, 2):
                        for y3 in range(-1, 2):
                            movingPoints = np.array([[1, 1], [Fp_camera_1.shape[1], 1], [1, Fp_camera_1.shape[0]], [Fp_camera_1.shape[1], Fp_camera_1.shape[0]]])
                            fixedPoints = movingPoints + np.array([[step * x1, step * y1], [step * x2, step * y2], [step * x3, step * y3], [0, 0]]) + transform_center_point

                            tform = cv2.getPerspectiveTransform(np.float32(movingPoints), np.float32(fixedPoints))
                            Noisex_t = cv2.warpPerspective(Noisex_ref, tform, (Noisex_ref.shape[1], Noisex_ref.shape[0]))

                            try:
                                C = cv2.matchTemplate(Fp_camera_1, Noisex_t, method=cv2.TM_CCOEFF_NORMED)
                                res = np.max(C)
                            except:
                                print('PCE calculation error')
                                res = 0

                            new_transform = [frame_id, res, 0, step * x1, step * y1, step * x2, step * y2, step * x3, step * y3, 0, 0]
                            new_transform[3:11] = new_transform[3:11] + transform_center[0, 3:11]
                            transform.append(new_transform)

    return transform

def find_Result(videoPath, save_name, pce_sub, nsub, pce_glob, isWeightExist):
    all_transforms = sio.loadmat(os.path.join(save_name, 'all_transforms.mat'))
    transform_2 = all_transforms['transform_2']
    fp = sio.loadmat(os.path.join(save_name, 'fp.mat'))
    best20 = find_best_20(transform_2)

    frames = [os.path.join(videoPath, file) for file in os.listdir(videoPath) if file.endswith('.mat') and file != 'db.mat']
    crop_size = 250
    ncc_range = 65
    y_size = fp['cropped_prnu'].shape[0] // 2
    x_size = fp['cropped_prnu'].shape[1] // 2

    limit = min(len(frames), np.max(best20[:, 0]))

    for i in range(limit):
        best20 = createSubBlocks(best20, i, frames, pce_glob, fp['cropped_prnu'])

    Fp_camera_1 = fp['cropped_prnu'][(y_size-crop_size-ncc_range):(y_size+crop_size+ncc_range), (x_size-crop_size-ncc_range):(x_size+crop_size+ncc_range)]
    aggregateFP = [0] * 5
    aggregateFPD = [1] * 5

    for i in range(min(len(frames), best20.shape[0])):
        candidateTransformDetail = best20[(i * 20):((i + 1) * 20), :]
        candidateTransformDetail[:, 17] = np.sum(candidateTransformDetail[:, 13:17] > pce_sub, axis=1)
        validTransforms = candidateTransformDetail[candidateTransformDetail[:, 17] > nsub, :]

        maxPceGlob = np.max(validTransforms, axis=0)
        if validTransforms.shape[0] > 4 and maxPceGlob[1] > pce_glob:
            Noisex = sio.loadmat(frames[i])['PRNU_image']
            weight = np.ones(Noisex.shape)
            if isWeightExist:
                weight = sio.loadmat(frames[i].replace('.mat', '_weight.mat'))['weight']

            Noisex_ref = Noisex[(y_size-crop_size):(y_size+crop_size), (x_size-crop_size):(x_size+crop_size)]
            weight_ref = weight[(y_size-crop_size):(y_size+crop_size), (x_size-crop_size):(x_size+crop_size)]

            for candidates in range(5):
                validTransform = validTransforms[-(candidates + 1), :]
                transform_point = validTransform[3:11].reshape((4, 2))
                movingPoints = np.array([[1, 1], [Fp_camera_1.shape[1], 1], [1, Fp_camera_1.shape[0]], [Fp_camera_1.shape[1], Fp_camera_1.shape[0]]])
                fixedPoints = movingPoints + transform_point

                tform = cv2.getPerspectiveTransform(np.float32(movingPoints), np.float32(fixedPoints))
                Noisex_t = cv2.warpPerspective(Noisex_ref, tform, (Noisex_ref.shape[1], Noisex_ref.shape[0]))
                Weight_t = cv2.warpPerspective(weight_ref, tform, (weight_ref.shape[1], weight_ref.shape[0]))

                C = cv2.matchTemplate(Fp_camera_1, Noisex_t, method=cv2.TM_CCOEFF_NORMED)
                fw_best_crop = 132 - np.unravel_index(np.argmax(C), C.shape)
                pad_crop_weight = np.pad(Weight_t, ((fw_best_crop[0],), (fw_best_crop[1],)), 'constant', constant_values=(1,))
                pad_crop_fp = np.pad(Noisex_t, ((fw_best_crop[0],), (fw_best_crop[1],)), 'constant', constant_values=(0,))

                aggregateFP[candidates] = aggregateFP[candidates] + pad_crop_fp
                aggregateFPD[candidates] = aggregateFPD[candidates] + pad_crop_weight

    for candidates in range(5):
        sio.savemat(os.path.join(save_name, f'aggregate{candidates}.mat'), {'aggregateFP': aggregateFP[candidates], 'aggregateFPD': aggregateFPD[candidates]})

def result(visionPath):
    videos = [os.path.join(root, file) for root, _, files in os.walk(visionPath) for file in files if file.endswith('p')]

    for i, videoPath in enumerate(videos):
        if i == 139:
            continue
        videoResPath = os.path.join(os.path.dirname(videoPath), os.path.basename(videoPath))

        all_transforms = sio.loadmat(os.path.join(videoResPath, 'all_transforms.mat'))
        transform_2 = all_transforms['transform_2']
        best20 = find_best_20(transform_2)

        frames = [os.path.join(videoPath, file) for file in os.listdir(videoPath) if file.endswith('.mat') and file != 'db.mat']
        if len(frames) < 30:
            continue

        f_results = sio.loadmat(os.path.join(videoResPath, 'fp.mat'))
        Fp_camera_1 = f_results['cropped_prnu']
        limit = min(len(frames), np.max(best20[:, 0]))

        for i in range(limit):
            best20 = createSubBlocks(best20, i, frames, 42, Fp_camera_1)

def find_best_20(transform_2):
    sorted_transforms = transform_2[np.argsort(transform_2[:, 1])]
    return sorted_transforms[-20:, :]

def createSubBlocks(best20, i, frames, pce_glob, Fp_camera_1):
    # Implement createSubBlocks function
    return best20

# Example usage
visionPath = 'path/to/vision'
isPreCalculated = False
isWeightExist = False

dude_Constrained3LevelHGS(visionPath, isPreCalculated, isWeightExist)


# import os
# import glob
# import time
# import scipy.io
#
# # 定义变量
# visionPath = 'path/to/vision/'
# isPreCalculated = 0
# isWeightExist = True
#
# # 获取视频列表
# videos = glob.glob(os.path.join(visionPath, '*/*/*p'))
#
# for i, video_path in enumerate(videos):
#     print(i)
#     if i == 138:  # 因为Python的索引从0开始
#         continue
#
#     start_time = time.time()
#     video_dir = os.path.dirname(video_path)
#     video_res_path = os.path.join(video_dir, os.path.basename(video_path))
#     print(f'Running video is {video_res_path}')
#
#     if isPreCalculated == 0:
#         cropped_prnu = None
#         # 加载fp.mat文件
#         fp_path = os.path.join(video_res_path, 'fp.mat')
#         if os.path.exists(fp_path):
#             mat_data = scipy.io.loadmat(fp_path)
#             cropped_prnu = mat_data.get('cropped_prnu')
#
#         calculate_Transforms(video_dir, cropped_prnu, video_res_path)
#
#     # 执行find_Result函数
#     find_Result(video_dir, video_res_path, 2, 2, 42, isWeightExist)
#
#     end_time = time.time()
#     print(f'Time taken: {end_time - start_time} seconds')


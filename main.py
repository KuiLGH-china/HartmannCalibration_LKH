import cv2
import numpy as np
from IPython.display import display, Image
from matplotlib import pyplot as plt
import os
from scipy.special import factorial as fac
from matplotlib.patches import Rectangle


def qgpu2sc(wp: np.ndarray,
            quality_map: np.ndarray = np.array([]),
            start: np.ndarray = np.array([]),
            num_of_quality_steps: int = 128,
            ):
    """
    Quality guided phase unwrapping with stack chain
    用堆叠链解图像质量为导向的包裹相位。

     Parameters
    ----------
        wp: `numpy.ndarray`
            包裹相位图
        quality_map: `numpy.ndarray`
            质量图
        start: `numpy.ndarray`
            解包相位的起始像素
        num_of_quality_steps: `int`
            质量步数
    Returns
    -------
        uwp: `numpy.ndarray`
            解包相位图
    """

    sz = wp.shape
    if quality_map.size == 0:
        quality_map = np.mat(np.hanning(sz[0])).T * np.hanning(sz[1])
    quality_thr = np.min(quality_map)
    mask = np.ones(sz, dtype=bool)

    # //! start_row start_col 最终取的是行列索引中的第一个元素
    if start.size == 0:
        start = np.where(quality_map == np.max(quality_map))
    # 起始像素的行和列。
    start_row, start_col = start
    if start_row.size > 1:
        start_row = int(start_row[0])
    if start_col.size > 1:
        start_col = int(start_col[0])
    min_q = np.min(quality_map)
    max_q = np.max(quality_map)
    if min_q != max_q:
        quality_map = np.int32(np.round(((quality_map - min_q) / (max_q - min_q)) * (num_of_quality_steps - 1)) + 1)
        if quality_thr >= min_q:
            quality_thr = np.round(((quality_thr - min_q) / (max_q - min_q)) * (num_of_quality_steps - 1)) + 1
        elif quality_thr < min_q:
            quality_thr = 1
    else:
        quality_map = np.int32(quality_map / min_q)
        quality_thr = 1

    # stack_chain:"堆栈链"
    stack_chain = np.int32(np.zeros(num_of_quality_steps + 1, ))
    uwg_row = np.zeros((wp.size,), dtype=int)
    uwg_col = np.zeros((wp.size,), dtype=int)
    uwd_row = np.zeros((wp.size,), dtype=int)
    uwd_col = np.zeros((wp.size,), dtype=int)
    stack_n = np.zeros((wp.size,))
    uwp = np.zeros_like(wp)
    path_map = np.zeros_like(wp)
    # queued_flag:排队标志
    queued_flag = np.zeros(sz, dtype=bool)
    quality_max = int(quality_map[start_row, start_col])
    stack_chain[quality_max] = 1
    pointer = 1
    unwr_order = 0
    uwd_row[stack_chain[quality_max]] = start_row
    uwd_col[stack_chain[quality_max]] = start_col
    uwg_row[stack_chain[quality_max]] = start_row
    uwg_col[stack_chain[quality_max]] = start_col

    path_map[start_row, start_col] = 1
    queued_flag[start_row, start_col] = True
    # Set unwrapping phase as wrapped phase for the starting point.
    # 将起始点的解包相位设置为包裹相位。
    uwp[start_row, start_col] = wp[start_row, start_col]
    # When quality_max is higher than quality_thr, flood fill.
    # 当 quality_max 大于 quality_thr 时，进行泛洪填充。

    # 1.进入一个循环，只要当前的最大质量值 quality_max 大于等于预设的阈值 quality_thr，就会一直进行解包。
    while quality_max >= quality_thr:
        # If stack_chain in quality_max level is currently empty, go to
        # quality_max-1 level.
        # 2.在每一轮循环中，它会检查当前质量级别 quality_max 对应的堆栈链 stack_chain 是否为空，如果为空，则将质量级别降低到 quality_max - 1。
        if stack_chain[quality_max] == 0:
            quality_max = quality_max - 1
        else:
            # Unwrap current point.解包当前点。
            # 3.如果当前堆栈链不为空，则会执行解包操作。首先，根据堆栈链中记录的位置信息，找到当前待解包点的位置，并计算其周围的相位差。
            uwdrow = int(uwd_row[stack_chain[quality_max]])
            uwdcol = int(uwd_col[stack_chain[quality_max]])
            a = uwp[uwdrow, uwdcol]

            uwgrow = int(uwg_row[stack_chain[quality_max]])
            uwgcol = int(uwg_col[stack_chain[quality_max]])
            b = wp[uwgrow, uwgcol]

            uwp[uwgrow, uwgcol] = b - 2 * np.pi * round((b - a) / (2 * np.pi))

            # Temporal row and column of the unwrapping point.
            # 解包点的临时行和列。
            # 4.接着，根据解包点的相位差更新其解包后的相位值，并记录解包点的位置信息，更新路径图以及解包顺序。
            temp_row = int(uwg_row[stack_chain[quality_max]])
            temp_col = int(uwg_col[stack_chain[quality_max]])

            # Update path_map.
            path_map[temp_row, temp_col] = unwr_order
            unwr_order = unwr_order + 1
            stack_chain[quality_max] = stack_n[stack_chain[quality_max]]
            # 5.然后，根据解包点的位置，检查其周围的相邻点，将符合条件的相邻点加入到队列中进行下一轮的解包。如果相邻点的质量值高于当前最大质量值，更新最大质量值。
            if (temp_row > 0):
                # Check unwrapping state and mask validity.
                # 检查解包状态和掩码的有效性。
                if (~queued_flag[temp_row - 1, temp_col]) \
                        and (mask[temp_row - 1, temp_col]):

                    # upper:(row-1,col)
                    # 上方：(行-1，列)
                    uwg_row[pointer] = int(temp_row - 1)
                    uwg_col[pointer] = int(temp_col)
                    uwd_row[pointer] = int(temp_row)
                    uwd_col[pointer] = int(temp_col)

                    # Push stack_chain to the stack_n at pointer.
                    # 将stack_chain推送到指针处的stack_n。
                    stack_n[pointer] = stack_chain[quality_map[uwg_row[pointer],
                    uwg_col[pointer]]]
                    # Push pointer to stack_chain.
                    # 将指针推送到stack_chain。
                    stack_chain[quality_map[uwg_row[pointer], uwg_col[pointer]]] = pointer

                    # If the quality value of pushed point is bigger than the
                    # current quality_max value, set the quality_max as the
                    # quality value of pushed point.
                    # # 如果推入点的质量值大于当前的quality_max值，则将quality_max设置为推入点的质量值。
                    if quality_map[uwg_row[pointer], uwg_col[pointer]] > quality_max:
                        quality_max = quality_map[uwg_row[pointer], uwg_col[pointer]]
                    # Queue the point.
                    # 将该点入队
                    queued_flag[uwg_row[pointer], uwg_col[pointer]] = True
                    # pointer++.
                    pointer = pointer + 1

            # the nether neighboring point: (row+1,col)
            # 下方相邻点：(行+1，列)
            # Check dimensional validity.
            # 检查维度的有效性
            if (temp_row < sz[0] - 1):
                # 检查解包状态和掩膜的有效性。
                if (~queued_flag[temp_row + 1, temp_col]) \
                        and (mask[temp_row + 1, temp_col]):
                    # 下方:(行+1,列)
                    uwg_row[pointer] = int(temp_row + 1)
                    uwg_col[pointer] = int(temp_col)
                    uwd_row[pointer] = int(temp_row)
                    uwd_col[pointer] = int(temp_col)

                    # 将stack_chain推入到指针处的stack_n中。
                    stack_n[pointer] = stack_chain[quality_map[uwg_row[pointer],
                    uwg_col[pointer]]]
                    # 将指针推入到stack_chain中。
                    stack_chain[quality_map[uwg_row[pointer],
                    uwg_col[pointer]]] = pointer

                    # 如果推入的点的质量值大于当前的quality_max值，则将quality_max设置为推入点的质量值。
                    if quality_map[uwg_row[pointer], uwg_col[pointer]] > quality_max:
                        quality_max = quality_map[uwg_row[pointer], uwg_col[pointer]]

                    # 将点加入队列。
                    queued_flag[uwg_row[pointer], uwg_col[pointer]] = True
                    # 指针+1。
                    pointer = pointer + 1

            # 左侧邻居点：(行，列-1)
            # 检查维度的有效性。
            if (temp_col > 0):
                # 检查解包状态和掩膜的有效性。
                if (~queued_flag[temp_row, temp_col - 1]) \
                        and (mask[temp_row, temp_col - 1]):

                    # 左侧:(行，列-1)
                    uwg_row[pointer] = int(temp_row)
                    uwg_col[pointer] = int(temp_col - 1)
                    uwd_row[pointer] = int(temp_row)
                    uwd_col[pointer] = int(temp_col)

                    # 将stack_chain推入到指针处的stack_n中。
                    stack_n[pointer] = stack_chain[quality_map[uwg_row[pointer],
                    uwg_col[pointer]]]
                    # 将指针推入到stack_chain中。
                    stack_chain[quality_map[uwg_row[pointer],
                    uwg_col[pointer]]] = pointer

                    # 如果推入的点的质量值大于当前的quality_max值，则将quality_max设置为推入点的质量值。
                    if quality_map[uwg_row[pointer], uwg_col[pointer]] > quality_max:
                        quality_max = quality_map[uwg_row[pointer], uwg_col[pointer]]

                    # 将点加入队列。
                    queued_flag[uwg_row[pointer], uwg_col[pointer]] = True
                    # 指针+1。
                    pointer = pointer + 1

            # 右侧邻居点：(行，列+1)
            # 检查维度的有效性。
            if (temp_col < sz[1] - 1):
                # 检查解包状态和掩膜的有效性。
                if (~queued_flag[temp_row, temp_col + 1]) \
                        and (mask[temp_row, temp_col + 1]):

                    # 右侧:(行，列+1)
                    uwg_row[pointer] = int(temp_row)
                    uwg_col[pointer] = int(temp_col + 1)
                    uwd_row[pointer] = int(temp_row)
                    uwd_col[pointer] = int(temp_col)

                    # 将stack_chain推入到指针处的stack_n中。
                    stack_n[pointer] = stack_chain[quality_map[uwg_row[pointer],
                    uwg_col[pointer]]]
                    # 将指针推入到stack_chain中。
                    stack_chain[quality_map[uwg_row[pointer],
                    uwg_col[pointer]]] = pointer

                    # 如果推入的点的质量值大于当前的quality_max值，则将quality_max设置为推入点的质量值。
                    if quality_map[uwg_row[pointer], uwg_col[pointer]] > quality_max:
                        quality_max = quality_map[uwg_row[pointer], uwg_col[pointer]]

                    # 将点加入队列。
                    queued_flag[uwg_row[pointer], uwg_col[pointer]] = True
                    # 指针+1。
                    pointer = pointer + 1
        # 不断重复这个过程，直到所有符合条件的点都被解包完毕或者最大质量值低于设定的阈值为止
        # 1.该点尚未被解包过，即其在解包路径图中的值为零。
        # 2.该点在掩膜范围内，即该点的位置在图像边界内，不会超出图像范围。
        # 3.该点的质量值符合解包的要求，即质量值大于当前的最小质量阈值，以确保解包过程中选择的是质量较高的相位点。

    # path_map=(unwr_order-path_map)*mask;
    return uwp


def crop_image_with_fringe_orders(img: np.ndarray,
                                  fringe_orders_u: np.ndarray,
                                  fringe_orders_v: np.ndarray,
                                  order_u: int = 0,
                                  order_v: int = 1):
    """
    Crop image with fringe orders
    使用条纹序号裁剪图像

    参数
    ----------
        img: `numpy.ndarray`
            要计算质心的图像
        fringe_orders_u: `numpy.ndarray`
            u方向的条纹序号
        fringe_orders_v: `numpy.ndarray`
            v方向的条纹序号
        order_u: `int`
            u方向要处理的序号
        order_v: `int`
            v方向要处理的序号
    返回
    -------
        sub_img: `numpy.ndarray`
            裁剪后的子图像
        min_u_in_order_mask: `int`
            序号掩码中的最小u值
        min_v_in_order_mask: `int`
            序号掩码中的最小v值
        u2d_sub_img: `numpy.ndarray`
            子图像的u坐标
        v2d_sub_img: `numpy.ndarray`
            子图像的v坐标
        avg_int_sub_img: `float`
            子图像的平均强度
    """
    nv_img, nu_img = img.shape
    u2d_img, v2d_img = np.meshgrid(np.arange(nu_img), np.arange(nv_img))

    order_mask = np.logical_and(fringe_orders_u == order_u, fringe_orders_v == order_v)

    # 从二维网格中提取出满足条件的像素的水平和垂直坐标，分别存储在 u2d_in_order_mask 和 v2d_in_order_mask 中。
    u2d_in_order_mask = u2d_img[order_mask == 1]
    v2d_in_order_mask = v2d_img[order_mask == 1]

    # 计算满足条件的像素在 u 和 v 方向上的最小和最大坐标，用于确定裁剪子图像的范围。
    min_u_in_order_mask, max_u_in_order_mask = np.min(u2d_in_order_mask), np.max(u2d_in_order_mask)
    min_v_in_order_mask, max_v_in_order_mask = np.min(v2d_in_order_mask), np.max(v2d_in_order_mask)

    # 从原始图像中裁剪出子图像 sub_img
    sub_img = img[min_v_in_order_mask:max_v_in_order_mask + 1, min_u_in_order_mask:max_u_in_order_mask + 1].copy()
    # 生成对应的子图像掩码 sub_mask
    sub_mask = order_mask[min_v_in_order_mask:max_v_in_order_mask + 1,
               min_u_in_order_mask:max_u_in_order_mask + 1].copy()

    # 根据子图像的灰度值最大值位置，确定子图像的半径范围，并根据该范围对子图像进行圆形裁剪。
    u2d_sub_img, v2d_sub_img = np.meshgrid(np.arange(sub_img.shape[1]), np.arange(sub_img.shape[0]))
    sub_img_max_v, sub_img_max_u = np.where(sub_img == np.nanmax(sub_img * sub_mask))
    mask_aera = np.sum(order_mask)
    r2d_sub_img = ((u2d_sub_img - int(sub_img_max_u[0])) ** 2 + (v2d_sub_img - int(sub_img_max_v[0])) ** 2) ** 0.5
    r_thr = (mask_aera / np.pi) ** 0.5
    r_mask = r2d_sub_img < r_thr
    sub_img[~r_mask] = 0

    # Calculate the average intensity of sub_img
    avg_int_sub_img = np.sum(sub_img) / np.sum(r_mask)

    return sub_img, min_u_in_order_mask, min_v_in_order_mask, u2d_sub_img, v2d_sub_img, avg_int_sub_img


def calculate_centroid(img: np.ndarray,
                       u2d_img: np.ndarray = None,
                       v2d_img: np.ndarray = None):
    """
    Calculate the centroid

    Parameters
    ----------
        img: `numpy.ndarray`
            The image to calcualte the centroid
        u2d_img: `numpy.ndarray`
            The u coordinates
        v2d_img: `numpy.ndarray`
            The v coordinates
    Returns
    -------
        centroid_u: `numpy.ndarray`
            The u coordinate of the centroid
        centroid_v: `numpy.ndarray`
            The v coordinate of the centroid
    """
    if (u2d_img is None) or (v2d_img is None):
        nv_img, nu_img = img.shape
        u2d_img, v2d_img = np.meshgrid(np.arange(nu_img), np.arange(nv_img))
    img_nansum = np.nansum(img)
    u2d_img_nansum, v2d_img_nansum = np.nansum(img * u2d_img), np.nansum(img * v2d_img)

    centroid_u, centroid_v = u2d_img_nansum / img_nansum, v2d_img_nansum / img_nansum
    return (centroid_u, centroid_v)


def nans(shape: tuple, dtype: type = np.float64):
    """
    Initialize array with numpy.nan

    Parameters
    ----------
        shape: `tuple`
            The shape of the array
        dtype: `type`
            The data type of the array
    Returns
    -------
        array: `numpy.ndarray`
            The array with numpy.nan

    """
    array = np.empty(shape, dtype)
    array.fill(np.nan)
    return array


def process_the_img(filename: str,
                    nu_detector: float = 512,  # 垂直像素数
                    nv_detector: float = 512,  # 水平像素数
                    upsampling: int = 2,  # 上采样倍率
                    pixel_depth: int = 14  # 像素深度
                    ):
    intensity = np.loadtxt(filename)

    # 上采样（插值）
    map_hr = np.flipud(intensity.reshape(nv_detector * upsampling, nu_detector * upsampling))
    temp_map_hr = map_hr.reshape(nv_detector, upsampling, nu_detector, upsampling)
    intensity_map = temp_map_hr.mean(axis=3).mean(axis=1)  # Binning

    # 像素深度归一化及确保像素深度在指定范围内
    img = np.floor((intensity_map - intensity_map.min()) / intensity_map.ptp() * (2 ** pixel_depth - 1))
    img[img > 2 ** pixel_depth - 1] = 2 ** pixel_depth - 1
    detector_image = img.astype(np.uint16)

    # 转换成更高精度的浮点数。
    img_float = detector_image.astype(np.float64)

    return img_float


def calculate_wrapped_phase(spectrum, u2d_img, v2d_img, min_fringe_number, is_u=True):
    # 给频谱采样
    v0, u0 = spectrum.shape[0] / 2, spectrum.shape[1] / 2
    un2d, vn2d = u2d_img - u0, v2d_img - v0

    # 频谱振幅
    amp = np.abs(spectrum)

    # 由振幅来制作索引
    rn2d = (un2d ** 2 + vn2d ** 2) ** 0.5
    r_mask = rn2d > min_fringe_number
    uv_mask = (vn2d < un2d) & (vn2d > -un2d) \
        if is_u == True else (vn2d > un2d) & (vn2d > -un2d)
    ruv_mask = r_mask & uv_mask
    filtered_amp = amp * ruv_mask
    idx = np.argmax(filtered_amp)  # 返回最大值的索引
    vc, uc = np.unravel_index(idx, spectrum.shape)

    # 检查是否需要使用第二谐波的一半频率来代替检测到的第一谐波。
    uc_half, vc_half = int(np.round((uc - u0) / 2 + u0)), int(np.round((vc - v0) / 2 + v0))
    if filtered_amp[vc, uc] < 3 * filtered_amp[vc_half, uc_half]:
        uc, vc = uc_half, vc_half

    # 低通滤波
    unc, vnc = uc - u0, vc - v0
    r_thr = rn2d[vc, uc] / 4
    rc2d = ((un2d - unc) ** 2 + (vn2d - vnc) ** 2) ** 0.5
    mask = rc2d < r_thr
    y = spectrum * mask

    # 计算包裹相位和振幅
    filtered_complex_img = np.fft.ifft2(np.fft.ifftshift(y))
    wrapped_phase = np.angle(filtered_complex_img)
    amplitude = np.abs(filtered_complex_img)

    return wrapped_phase, amplitude


def calcule_the_changes(img,
                        starting_pixel=None,  # 起始像素
                        block_size=31,  # 自适应领域大小
                        pixel_size=1.48e-6,  # 像素大小
                        area_thr=1,  # 面积阈值
                        min_fringe_number: int = 8,  # 最小条纹阶数（滤波）
                        ratio_thr: float = 0.05,  # 比率阈值
                        centroid_power: float = 1.7,  # 计算质心时的加权因子
                        grid_period: float = 20.e-6  # 条纹间距
                        ):
    # 1.确定采样间隔（简历像素坐标系）

    nv_img, nu_img = img.shape
    u2d_img, v2d_img = np.meshgrid(np.arange(nu_img), np.arange(nv_img))
    x2d_img, y2d_img = u2d_img * pixel_size, v2d_img * pixel_size

    # 2.确定斑点标签

    # 高斯模糊化。
    tmp = (img / img.max() * (2 ** 8 - 1))
    img_8bit = tmp.astype(np.uint8)
    img_8bit_blur = cv2.GaussianBlur(img_8bit, (5, 5), 0)

    # 自适应阈值化（二值化）
    img_bw = cv2.adaptiveThreshold(img_8bit_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size,
                                   -1)

    # 区域分割，标记可能的点
    num_of_rois, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bw)
    area = stats[:, 4]
    num_of_valid_rois = np.sum(area >= area_thr) - 1  # 去除最大面积的连通区域（背景）
    roi_idx_of_valid_rois = np.zeros(num_of_valid_rois)
    num_of_valid_rois = 0
    for roi_idx in range(1, num_of_rois):
        if area[roi_idx] >= area_thr:  # Exclude the background
            roi_idx_of_valid_rois[num_of_valid_rois] = roi_idx
            num_of_valid_rois = num_of_valid_rois + 1
        else:
            labels[labels == roi_idx] = 0

    # 3.解相位

    # 快速傅里叶变换，滤波，计算包裹相位及振幅
    x = np.fft.fft2(img)
    spectrum = np.fft.fftshift(x)
    wpu, ampu = calculate_wrapped_phase(spectrum, u2d_img, v2d_img, min_fringe_number, is_u=True)
    wpv, ampv = calculate_wrapped_phase(spectrum, u2d_img, v2d_img, min_fringe_number, is_u=False)

    # 计算质量图，计算起始像素点
    quality_map = (ampu + ampv) / 2
    quality_img = np.uint8(
        255 * (quality_map - np.min(quality_map)) / (np.max(quality_map) - np.min(quality_map)))  # 归一化
    quality_img_bw = cv2.threshold(quality_img, 0, 1, cv2.THRESH_OTSU)[1]  # 质量图二值化
    if starting_pixel is None:
        centroid_u, centroid_v = calculate_centroid(quality_img_bw, u2d_img, v2d_img)
        area = np.sum(quality_img_bw * 1)
        r_thr = ((area / np.pi) ** 0.5) / 2  # 等面积圆半径的一半
        # 设置相位搜索掩码
        r2d = ((u2d_img - centroid_u) ** 2 + (v2d_img - centroid_v) ** 2) ** 0.5
        searching_mask = r2d < r_thr  # Only search inside the central area

        # 找到一起始像素点
        quality = quality_map[searching_mask]
        min_quality = np.min(quality)
        v_min_quality, u_min_quality = np.where(quality_map == min_quality)
        starting_pixel = np.array([v_min_quality, u_min_quality])

    # 4.解包裹

    uwpu = qgpu2sc(wpu, quality_map, starting_pixel)
    uwpv = qgpu2sc(wpv, quality_map, starting_pixel)

    # 离散（计算条纹数）
    fringe_orders_u, fringe_orders_v = np.round((uwpu - wpu) / (np.pi * 2)), np.round((uwpv - wpv) / (np.pi * 2))

    # 5. 将斑点与条纹数对应

    orders_of_labelled_rois = np.zeros((num_of_valid_rois, 2))
    for idx_of_valid_roi in range(num_of_valid_rois):
        idx = labels == roi_idx_of_valid_rois[idx_of_valid_roi]
        orders_of_labelled_rois[idx_of_valid_roi, 0] = int(np.median(fringe_orders_u[idx]))
        orders_of_labelled_rois[idx_of_valid_roi, 1] = int(np.median(fringe_orders_v[idx]))
    complex_orders = orders_of_labelled_rois[:, 0] + orders_of_labelled_rois[:, 1] * 1j
    unique_orders = np.unique(complex_orders)

    # 确定一个裁剪图像像素值大小的标准
    example_order_u, example_order_v = 2, 3
    sub_img_example = crop_image_with_fringe_orders(img, fringe_orders_u, fringe_orders_v, order_u=example_order_u,
                                                    order_v=example_order_v)[0]
    sum_of_sub_img_example = np.sum(sub_img_example)

    # 6. 计算位移量

    # 位移量
    cx1d = nans(unique_orders.shape)
    cy1d = nans(unique_orders.shape)

    for nIdx in range(unique_orders.size):
        order_u, order_v = np.real(unique_orders[nIdx]), np.imag(unique_orders[nIdx])
        if not (order_u == 0 and order_v == 0) and not (np.isnan(order_u) or np.isnan(order_v)):
            # 4.2.1. Crop image
            # 4.2.1. 裁剪图像
            sub_img, min_u_in_order_mask, min_v_in_order_mask, \
                u2d_sub_img, v2d_sub_img, ai_sub_img = \
                crop_image_with_fringe_orders(img,
                                              fringe_orders_u,
                                              fringe_orders_v,
                                              order_u,
                                              order_v)
            if np.sum(sub_img) / sum_of_sub_img_example >= ratio_thr:
                powered_sub_img = sub_img ** centroid_power  # 计算加权图像：Calculate powered image
                u0d_cen_in_sub_img, v0d_cen_in_sub_img = calculate_centroid(powered_sub_img, u2d_sub_img, v2d_sub_img)
                # 最终质心坐标
                u0d_centroid, v0d_centroid = u0d_cen_in_sub_img + min_u_in_order_mask, \
                                             v0d_cen_in_sub_img + min_v_in_order_mask
                # Calculate the reference aperture centers
                u0d_aperture_center = img.shape[1] / 2 - 0.5 + (grid_period / pixel_size) * order_u
                v0d_aperture_center = img.shape[0] / 2 - 0.5 + (grid_period / pixel_size) * order_v
                # Calculate the spot location changes
                du0d, dv0d = u0d_centroid - u0d_aperture_center, v0d_centroid - v0d_aperture_center
                # 位移量
                cx1d[nIdx] = du0d
                cy1d[nIdx] = dv0d

    # 组装矩阵
    edge_exclusion: int = 1  # 假设边缘处的波前收到衍射影响，故而将其排除掉。
    unique_orders_u, unique_orders_v = unique_orders.real, unique_orders.imag
    min_order_u = int(np.nanmin(unique_orders_u[:])) + edge_exclusion
    max_order_u = int(np.nanmax(unique_orders_u[:])) - edge_exclusion
    min_order_v = int(np.nanmin(unique_orders_v[:])) + edge_exclusion
    max_order_v = int(np.nanmax(unique_orders_v[:])) - edge_exclusion
    nu_wfr, nv_wfr = max_order_u - min_order_u + 1, max_order_v - min_order_v + 1
    cu1d = nans((nv_wfr, nu_wfr))
    cv1d = nans((nv_wfr, nu_wfr))

    for order_v in range(min_order_v, max_order_v + 1):
        for order_u in range(min_order_u, max_order_u + 1):
            m, n = order_v - min_order_v, order_u - min_order_u
            idx = (unique_orders_u == order_u) & (unique_orders_v == order_v)
            if np.any(idx) == True:
                cu1d[m, n] = cx1d[idx]
                cv1d[m, n] = cy1d[idx]

    # 位移量
    cu1d = nans((nv_wfr, nu_wfr))
    cv1d = nans((nv_wfr, nu_wfr))

    return cu1d, cv1d


def final_calculate(imga, imgb, delta_sx, delta_sy, f):
    # 位移前
    img_A = process_the_img(imga)
    cu1d_A, cv1d_A = calcule_the_changes(img_A)

    # 位移后
    img_B = process_the_img(imgb)
    cu1d_B, cv1d_B = calcule_the_changes(img_B)

    delta_Cx = cu1d_A - cu1d_B
    delta_Cy = cv1d_A - cv1d_B
    Lx = delta_Cx * f / delta_sx
    Ly = delta_Cy * f / delta_sy
    return Lx, Ly


# ########################################################################################################################
# data_path = 'data_example_21'
# hartmanngram_fn = 'ex21_res_int_pr_se.dat'
# # 将图像数据转换成png图像并保存在特定路径。
# hartmanngram_png_fn = hartmanngram_fn + '.png'
# img1 = os.path.join(data_path, hartmanngram_fn)
# Lx, Ly = final_calculate(img1, img1, 1, 1, 5)
# print(Lx, Ly)
#
# plt.imshow(Ly, cmap='jet')
# plt.show()
#
# plt.imshow(Lx, cmap='jet')
# plt.show()




# ================= 核心计算函数 =================
def calculate_metrics_for_array(binary_mask, pixel_size_x, pixel_size_y):
    """
    基于二值掩膜计算6个景观指标
    binary_mask: 0/1 矩阵 (1=林地)
    pixel_size_x, pixel_size_y: 像元的分辨率
    """
    # 1. 斑块识别 (8邻域)
    labeled_array = label(binary_mask, connectivity=2)
    props = regionprops(labeled_array)
    num_patches = len(props)

    # 获取景观总面积 (通常指这个网格的总面积，或者裁剪出来的有效区域)
    # 这里我们假设网格是完整的，用掩膜的总像素数计算景观面积
    total_landscape_pixels = binary_mask.size
    landscape_area_ha = (total_landscape_pixels * pixel_size_x * pixel_size_y) / 10000.0

    # 如果没有林地
    if num_patches == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # 2. 提取斑块属性
    # 面积 (m2)
    patch_areas_m2 = np.array([p.area for p in props]) * pixel_size_x * pixel_size_y
    # 面积 (ha)
    patch_areas_ha = patch_areas_m2 / 10000.0
    # 周长 (m)
    patch_perimeters_m = np.array([p.perimeter for p in props]) * pixel_size_x

    # --- 指标计算 ---

    # 1. PD: 斑块密度 (个/100公顷)
    # 景观生态学中标准单位通常是 Number per 100 ha
    if landscape_area_ha > 0:
        PD = (num_patches / landscape_area_ha) * 100
    else:
        PD = 0

    # 2. ED: 边缘密度 (米/公顷)
    total_edge_m = np.sum(patch_perimeters_m)
    if landscape_area_ha > 0:
        ED = total_edge_m / landscape_area_ha
    else:
        ED = 0

    # 3. MPA: 平均斑块面积 (公顷)
    MPA = np.mean(patch_areas_ha)

    # 4. LSI: 景观形状指数
    # Class-level LSI = 0.25 * Total_Edge / Sqrt(Total_Class_Area)
    total_class_area_m2 = np.sum(patch_areas_m2)
    if total_class_area_m2 > 0:
        LSI = (0.25 * total_edge_m) / np.sqrt(total_class_area_m2)
    else:
        LSI = 0  # 或者是 1，视定义而定，无林地时通常为0

    # 5. ENN_MN: 平均欧氏邻近距离 (米)
    if num_patches > 1:
        centroids = np.array([p.centroid for p in props])
        # 将像素坐标转为米 (注意 centroid 返回的是 (row, col))
        centroids[:, 0] *= pixel_size_y
        centroids[:, 1] *= pixel_size_x

        dist_matrix = squareform(pdist(centroids))
        np.fill_diagonal(dist_matrix, np.inf)
        min_dists = np.min(dist_matrix, axis=1)
        ENN_MN = np.mean(min_dists)
    else:
        ENN_MN = 0.0

    # 6. COHESION: 斑块结合度 (0-100)
    # 简化计算，注意单位一致性，使用像素单位计算比率
    p_pixels = np.array([p.perimeter for p in props])
    a_pixels = np.array([p.area for p in props])
    Z = total_landscape_pixels

    numerator = np.sum(p_pixels)
    denominator = np.sum(p_pixels * np.sqrt(a_pixels))

    if denominator != 0:
        term1 = 1 - (numerator / denominator)
        term2 = 1 - (1 / np.sqrt(Z))
        COHESION = term1 * (1 / term2) * 100
    else:
        COHESION = 0.0

    return PD, ED, MPA, LSI, ENN_MN, COHESION
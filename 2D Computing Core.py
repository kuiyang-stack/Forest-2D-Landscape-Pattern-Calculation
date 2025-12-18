import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
import numpy as np
import pandas as pd
from shapely.geometry import shape, box
from scipy import ndimage
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------------
# 连通性与聚集性指标函数
# -------------------------------
def compute_enn_mn(patch_polygons):
    """计算平均最近邻距离 ENN_MN"""
    if len(patch_polygons) < 2:
        return 0.0
    centroids = np.array([[p.centroid.x, p.centroid.y] for p in patch_polygons])
    tree = cKDTree(centroids)
    distances, _ = tree.query(centroids, k=2)  # 最近邻距离 k=2，第一个是自己
    enn_mn = np.mean(distances[:, 1])
    return float(enn_mn)

def compute_cohesion_raster_fixed(patch_areas, pixel_area):
    """
    修正 COHESION，基于像元数量
    patch_areas: 每个斑块面积（平方米）
    pixel_area: 每个像元面积（平方米）
    返回 0~100
    """
    if len(patch_areas) == 0:
        return 0.0
    patch_pixels = np.array(patch_areas) / pixel_area
    max_perimeter = np.sum(4 * np.sqrt(patch_pixels))
    if max_perimeter == 0:
        return 0.0
    cohesion = (1 - len(patch_areas) / max_perimeter) * 100.0
    cohesion = max(0.0, min(cohesion, 100.0))
    return cohesion

# -------------------------------
# 核心景观指标计算函数
# -------------------------------
def calculate_landscape_metrics_full(grid_geometry, lucc_path, crop_value, verbose=False):
    """
    计算景观指标: ED, PD, MPA, LSI, ENN_MN, COHESION
    """
    try:
        if not grid_geometry.is_valid:
            grid_geometry = grid_geometry.buffer(0)

        with rasterio.open(lucc_path) as src:
            raster_bounds = box(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
            if not grid_geometry.intersects(raster_bounds):
                return {'ED': 0.0, 'PD': 0.0, 'MPA': 0.0, 'LSI': 0.0,
                        'ENN_MN': 0.0, 'COHESION': 0.0, 'has_cropland': False}
            try:
                out_image, out_transform = mask(src, [grid_geometry], crop=True, all_touched=True, filled=True)
            except Exception as e:
                if verbose:
                    print(f"  mask 裁剪失败: {e}")
                return {'ED': 0.0, 'PD': 0.0, 'MPA': 0.0, 'LSI': 0.0,
                        'ENN_MN': 0.0, 'COHESION': 0.0, 'has_cropland': False}
            nodata = src.nodata

        band = out_image[0].copy()
        if nodata is not None:
            band = np.where(band == nodata, 0, band)
        cropland = (band == crop_value).astype(np.uint8)
        if cropland.sum() == 0:
            return {'ED': 0.0, 'PD': 0.0, 'MPA': 0.0, 'LSI': 0.0,
                    'ENN_MN': 0.0, 'COHESION': 0.0, 'has_cropland': False}

        pixel_width = abs(out_transform.a)
        pixel_height = abs(out_transform.e)
        pixel_area = pixel_width * pixel_height
        landscape_area_ha = cropland.size * pixel_area / 10000.0

        structure = np.ones((3, 3), dtype=np.int8)
        labeled, n_patches = ndimage.label(cropland, structure=structure)

        patch_areas = []
        patch_polygons = []
        for geom, val in shapes(cropland, mask=cropland.astype(bool), transform=out_transform):
            if int(val) == 1:
                poly = shape(geom)
                patch_polygons.append(poly)
                patch_areas.append(poly.area)

        if len(patch_areas) == 0:
            return {'ED': 0.0, 'PD': 0.0, 'MPA': 0.0, 'LSI': 0.0,
                    'ENN_MN': 0.0, 'COHESION': 0.0, 'has_cropland': True}

        total_perimeter_m = np.sum([np.sqrt(a) * 4 for a in patch_areas])  # 用像元边长近似
        mean_patch_area_m2 = np.mean(patch_areas)

        # ED, PD, MPA, LSI
        ed_m_per_ha = total_perimeter_m / landscape_area_ha
        pd_per_100ha = (len(patch_areas) / landscape_area_ha) * 100
        mpa_ha = mean_patch_area_m2 / 10000.0
        lsi_index = (0.25 * total_perimeter_m) / np.sqrt(np.sum(patch_areas))

        # ENN_MN
        enn_mn = compute_enn_mn(patch_polygons)

        # COHESION
        cohesion = compute_cohesion_raster_fixed(patch_areas, pixel_area)

        return {
            'ED': round(ed_m_per_ha, 4),
            'PD': round(pd_per_100ha, 4),
            'MPA': round(mpa_ha, 4),
            'LSI': round(lsi_index, 4),
            'ENN_MN': round(enn_mn, 4),
            'COHESION': round(cohesion, 4),
            'has_cropland': True
        }

    except Exception as e:
        if verbose:
            print(f"  网格处理出错: {e}")
        return {'ED': 0.0, 'PD': 0.0, 'MPA': 0.0, 'LSI': 0.0,
                'ENN_MN': 0.0, 'COHESION': 0.0, 'has_cropland': False}


# -------------------------------
# 可视化函数
# -------------------------------
def create_visualization(grid_gdf_with_metrics, jenks_k=5):
    import warnings
    warnings.filterwarnings('ignore')
    use_jenks = True
    try:
        import mapclassify
    except Exception:
        use_jenks = False
        print("未安装 mapclassify，退回连续配色。")

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    valid_gdf = grid_gdf_with_metrics[grid_gdf_with_metrics['has_cropland']]

    metrics = ['ED', 'PD', 'MPA', 'LSI', 'ENN_MN', 'COHESION']
    cmaps = ['YlOrRd', 'Blues', 'Greens', 'Purples', 'Oranges', 'RdPu']

    for ax, metric, cmap in zip(axes.flatten(), metrics, cmaps):
        if len(valid_gdf) > 0:
            if use_jenks and valid_gdf[metric].nunique() > 1:
                valid_gdf.plot(column=metric, ax=ax, legend=True,
                               cmap=cmap, scheme='NaturalBreaks', k=jenks_k,
                               edgecolor='black', linewidth=0.1)
            else:
                valid_gdf.plot(column=metric, ax=ax, legend=True,
                               cmap=cmap, edgecolor='black', linewidth=0.1)
            ax.set_title(f'{metric} 指标')
    plt.tight_layout()
    out_png = 'complete_fragmentation_analysis_full.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"可视化已保存: {out_png}")



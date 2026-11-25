# Applications

Real-world domains where semantic segmentation is deployed.

---

## Medical Imaging

The original motivation for U-Net. Tumor segmentation, organ delineation, and radiology analysis.

| Application | Dataset | Task |
|------------|---------|------|
| Brain tumor | BraTS | Tumor core, edema, enhancing tumor |
| Cell tracking | Cell Tracking Challenge | Instance segmentation of cells |
| Organ segmentation | Various CT/MRI | Liver, kidney, lung delineation |

- Class imbalance is common (small lesion, large background)
- High precision required — false negatives are critical

**Datasets:**
- [BraTS (Brain Tumor Segmentation)](https://www.med.upenn.edu/cbica/brats/)
- [Medical Segmentation Decathlon](http://medicaldecathlon.com/)

**Papers:**
- [Ronneberger et al., 2015 — U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

---

## Autonomous Driving

Road, lane, vehicle, and pedestrian segmentation for scene understanding.

- Real-time inference required (30+ FPS)
- Multi-class: road, sidewalk, building, sky, car, pedestrian, etc.

**Datasets:**
- [Cityscapes](https://www.cityscapes-dataset.com/)
- [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/)
- [Mapillary Vistas](https://www.mapillary.com/dataset/vistas)

---

## Satellite Imagery

Land cover classification, urban planning, and environmental monitoring.

- Very high resolution images (10,000+ pixels)
- Often processed in tiles/patches
- Multi-spectral bands (visible + infrared)

**Datasets:**
- [DeepGlobe Land Cover Classification](https://competitions.codalab.org/competitions/18468)
- [SpaceNet](https://spacenet.ai/)

---

## Agriculture

Crop disease detection, weed identification, and yield estimation from drone/satellite imagery.

- Detecting diseased vs healthy plant regions
- Counting and localizing individual plants
- Precision agriculture applications

**Datasets:**
- [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset)
- [Sentinel-2 (ESA)](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)

# MultiZoo Image Processing

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Image Processing](https://img.shields.io/badge/Image%20Processing-numpy%2C%20PIL-lightgrey)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

Bu proje, birden çok hayvan türünün görüntülerini işlemek amacıyla hazırlanmış temel bir **image processing** çalışmasıdır. 
Veri ön işleme, filtreleme, boyutlandırma gibi işlemlerle birden çok sınıfa uygun hale getirme üzerine odaklanır.

---

## Özellikler

- Görüntü okuma ve ön işlem (resize, normalize)  
- Renk uzayı dönüşümleri (grayscale, HSV, vb.)  
- Filtre uygulamaları (blur, edge detection)  
- Görüntü sınıflandırma öncesi hazırlık (crop, augmentation)  
- Çoklu hayvan sınıfları için esnek pipeline

---


## Gereksinimler

Python 3.8 veya üstü. Önerilen kütüphaneler:

- `Pillow`>=9.0
- `numpy`>=1.20
- `matplotlib`>=3.4
- (Opsiyonel) `opencv-python`>=4.5 


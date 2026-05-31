"""Merkezi konfigürasyon — tüm algoritma ve yol sabitleri."""

# --- Ön işleme (CLAHE & Blur) ---
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID_SIZE = (8, 8)
MEDIAN_BLUR_KSIZE = 5
BLUR_THRESHOLD = 100.0

# --- WBC (Akyuvar) segmentasyonu ---
WBC_LOWER_PURPLE = [120, 30, 30]
WBC_UPPER_PURPLE = [175, 255, 255]
WBC_MORPH_KERNEL_SIZE = (5, 5)
WBC_MIN_AREA = 800
WBC_RECT_COLOR = (0, 255, 0)
WBC_RECT_THICKNESS = 4
WBC_FONT_SCALE = 0.8
WBC_FONT_THICKNESS = 2

# --- RBC (Alyuvar) segmentasyonu ---
ADAPTIVE_THRESH_BLOCK_SIZE = 11
ADAPTIVE_THRESH_C = 2
ADAPTIVE_THRESH_MAX_VAL = 255
RBC_MORPH_KERNEL_SIZE = (3, 3)
RBC_MORPH_OPEN_ITERATIONS = 2
RBC_DILATE_ITERATIONS = 2
DIST_TRANSFORM_MASK_SIZE = 5
WATERSHED_THRESH_COEFF = 0.1
WATERSHED_THRESH_MAX_VAL = 255
RBC_MIN_AREA = 30
RBC_CIRCLE_RADIUS = 3
RBC_CIRCLE_COLOR = (255, 0, 0)

# --- Kenar hücre filtresi (kesik/yarım hücreleri ele) ---
BORDER_MARGIN = 3

# --- Varsayılan yollar ---
DEFAULT_INPUT_DIR = "data/BCCD_Dataset-master/BCCD/JPEGImages"
DEFAULT_OUTPUT_REPORT = "output/kan_sayim_raporu.xlsx"
DEFAULT_PROCESSED_DIR = "output/processed_images"

# --- WBC kırpma (ML veri seti) ---
SAVE_CROPPED_WBC = True
CROPPED_DIR = "output/extracted_wbcs/"

# --- Klinik referans aralıkları (mikroskop alanı / görüntü başına) ---
REF_WBC_MIN = 5
REF_WBC_MAX = 15
REF_RBC_MIN = 200
REF_RBC_MAX = 300

# --- Loglama ---
LOG_DIR = "logs"
LOG_FILE = "logs/pipeline.log"

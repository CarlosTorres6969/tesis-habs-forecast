import glob, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from make_maps import _clear_water_score, _best_scene
import config as C

for folder, meta in C.REGIONS.items():
    key = meta["key"]
    tifs = sorted(glob.glob(os.path.join(C.DIR_IMAGENES, folder, "**", "*.tif"), recursive=True))
    if not tifs:
        print(f"{key}: sin imagenes")
        continue
    best = _best_scene(tifs)
    best_score = _clear_water_score(best)
    worst = min(tifs[:10], key=_clear_water_score)
    worst_score = _clear_water_score(worst)
    print(f"\n{key} ({len(tifs)} imgs):")
    print(f"  MEJOR  score={best_score:.0f} -> {os.path.basename(best)}")
    print(f"  PEOR   score={worst_score:.0f} -> {os.path.basename(worst)}")

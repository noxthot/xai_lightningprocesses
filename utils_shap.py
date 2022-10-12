import ccc

META_INFIX = "_meta"
SHAP_INFIX = "_shapval"

colname_meta_infix = lambda col: col.replace("_lvl", META_INFIX + "_lvl") if "_lvl" in col else col + META_INFIX
colname_shap_infix = lambda col: col.replace("_lvl", SHAP_INFIX + "_lvl") if "_lvl" in col else col + SHAP_INFIX

META_COLS_NOLVL = ['hour', 'day', 'month', 'year', 'flash', 'longitude', 'latitude']
META_COLS = list(set(META_COLS_NOLVL + ccc.TRAIN_COLS))
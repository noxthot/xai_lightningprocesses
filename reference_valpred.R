library(mgcv)
library(arrow)

MODEL <- "2022_02_22__reference"
MODEL_PATH <- sprintf("data/models/targetmode_1/%s", MODEL)

b <- readRDS(sprintf("%s/reference_model.rds", MODEL_PATH))
d <- data.frame(flash = b$y, output = b$fitted)
write_parquet(d, sink = sprintf("%s/val_predictions.parquet", MODEL_PATH))

message(" * FIN.")
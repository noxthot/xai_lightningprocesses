message(" * start time: ", Sys.time())

library(dplyr)
library(purrr)
library(arrow)
library(mgcv)

DATA_PATH <- "data/data_processed/datamode_3/dataparquet_2022_02_21/"
DATE <- gsub("-", "_", Sys.Date())
MODEL_PATH <- sprintf("data/models/targetmode_1/%s__reference", DATE)

COLUMNS <- c("longitude", "latitude", "flash", "topography", "t2m", "cbh", 
    "cswc2040", "cth", "cape", "cp", "wvc1020", "ishf", "mcc", "tcslw", 
    "hour", "day", "dayofyear", "month")

message(" * read training data")   ## TODO: This should work w/o map and bind_rows but w/ SubTreeFileSystem
fls <- list.files(DATA_PATH, recursive = TRUE, full.names = TRUE)
idx <- !grepl("(SUCCESS|year=2019)", fls)
fls <- fls[idx]
dtrain <- map(fls, read_parquet, col_select = all_of(COLUMNS)) %>% bind_rows

message(" * handle cases w/o clouds")
dtrain <- dtrain %>%
    mutate(existing_cloud = factor(is.na(cbh),
            levels = c(TRUE, FALSE),
	    labels = c("NO_CLOUD", "CLOUD")
	),
	cbh = if_else(existing_cloud == "CLOUD", cbh, 0),
	cth = if_else(existing_cloud == "CLOUD", cth, 0)
    )

message(" * set up formula")
f <- flash ~ s(longitude, latitude, hour) + s(topography) +
    s(dayofyear, bs = "ts") + existing_cloud +
    s(cape, bs = "ts") + s(cp, bs = "ts") +
    s(cswc2040, bs = "ts") + s(cth, bs = "ts") +
    s(ishf, bs = "ts") + s(mcc, bs = "ts") +
    s(t2m, bs = "ts") + s(tcslw, bs = "ts") +
    s(wvc1020, bs = "ts")

message(" * fit model")
b <- bam(f,
    family = "binomial", data = dtrain,
    discrete = TRUE, nthreads = 4L
)
message(" * store model")
saveRDS(b, file = sprintf("%s/reference_model.rds", MODEL_PATH))

message(" * read test data")    ## TODO: Resolve dublicated code
fls <- list.files(DATA_PATH, recursive = TRUE, full.names = TRUE)
idx <- grepl("year=2019", fls)
fls <- fls[idx]
dtest <- map(fls, read_parquet, col_select = all_of(COLUMNS)) %>% bind_rows

message(" * handle cases w/o clouds")
dtest <- dtest %>%
    mutate(existing_cloud = factor(is.na(cbh),
            levels = c(TRUE, FALSE),
	    labels = c("NO_CLOUD", "CLOUD")
	),
	cbh = if_else(existing_cloud == "CLOUD", cbh, 0),
	cth = if_else(existing_cloud == "CLOUD", cth, 0)
    )

message(" * predict on test data")
dtest[["fit"]] <- predict(b, newdata = dtest, se.fit = FALSE, type = "response")

dtest <- dtest %>%
    select(longitude, latitude, flash, hour, day, month, fit)

message(" * store test predictions")
write_parquet(dtest, sink = sprintf("%s/test_predictions.parquet", MODEL_PATH))

message(" * end time: ", Sys.time())
message(" * FIN.")


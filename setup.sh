curl --output delays.csv https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/ebf331a5-df6c-437c-9f35-8093587c679a
duckdb -c "COPY (SELECT * FROM read_csv_auto('delays.csv')) TO 'delays.parquet' (FORMAT PARQUET);"


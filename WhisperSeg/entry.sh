#!/bin/bash

# Retrieve environment variables
bucket_name="${bucket_name}"
prefix="${prefix}"
filename="${filename}"


src="s3://${bucket_name}/${filename}"
dest="./${filename}"


if [ ! -d "$prefix" ]; then
    aws s3 cp $src $dest 
fi
# Execute the main script with environment variables as arguments
python inference.py "$bucket_name" "$prefix" "$filename"

aws s3 cp "./${prefix}/activity_detection" "s3://${bucket_name}/${prefix}/activity_detection/" --recursive

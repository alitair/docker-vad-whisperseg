# Start from a EC2 base image
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.3.0-gpu-py311-cu121-ubuntu20.04-ec2

# Set up the working directory for WhisperSeg
WORKDIR /opt/ml/model/WhisperSeg

# Clone the WhisperSeg repository
COPY WhisperSeg/ /opt/ml/model/WhisperSeg/

# Install ffmpeg
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

    
# Install the WhisperSeg repository's own dependencies
RUN pip install --no-cache-dir -r /opt/ml/model/WhisperSeg/requirements.txt

# Ensure that the script is executable
RUN chmod +x /opt/ml/model/WhisperSeg/inference.py
RUN chmod +x /opt/ml/model/WhisperSeg/entry.sh

# Command to run your application
ENTRYPOINT ["./entry.sh"]




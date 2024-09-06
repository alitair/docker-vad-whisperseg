import os
import pandas as pd
import librosa
from model import WhisperSegmenterFast
import sys

# This function is called once when the model is loaded.
def model_fn(model_dir):
    # model_path = os.path.join(model_dir, "model")
    model = WhisperSegmenterFast("nccratliri/whisperseg-animal-vad-ct2")  # Initialize model without specifying device
    return model


# This function is called to perform inference on the input data.
def predict_fn(request_body, model):

    filename    = request_body['filename']

    names = process_filename(filename)
    
    audio, sr = librosa.load(filename, sr=None, mono=False)
    audio = audio[0:len(names)]

    segment_params = {
        'min_frequency': 0,
        'spec_time_step': 0.001,
        'min_segment_length': 0.005,
        'eps': 0.01,
        'num_trials': 3,
        'batch_size': 8
    }

    for name, audio in zip(names, audio):
        csv_save_path = os.path.join( request_body["subdirectory"] , f"{name}.csv")
        segment_and_save(audio, sr, csv_save_path, model, **segment_params)



# This function is called to format the output data for the response.
def output_fn(prediction, content_type):
    if content_type == 'application/json':
        return prediction
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def process_filename(file_path):
    filename  = os.path.basename(file_path)
    filename_without_ext = os.path.splitext(filename)[0]
    parts = filename_without_ext.split('-')
    processed_parts = parts[1:]
    return processed_parts

def segment_and_save(audio, sr, save_path, segmenter, **segment_params):
    print( f"Segmenting {save_path}")
    segments = segmenter.segment(audio, sr=sr, **segment_params)
    df = pd.DataFrame(segments)
    df.to_csv(save_path, index=False)

def main():

    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")

    request_body = {
        'bucket_name': sys.argv[1],
        'prefix'     : sys.argv[2],
        'filename'   : sys.argv[3]
    }
    print('bucket_name:', request_body['bucket_name'])
    print('prefix:'     , request_body['prefix'])
    print('filename:'   , request_body['filename'])
    
    # Check if the file exists
    if os.path.exists(request_body['filename']):
        print('filename:', request_body['filename'])
    else:
        print('File does not exist:', request_body['filename'])

    request_body["subdirectory"] = os.path.join(request_body['prefix'], "activity_detection")
    print("making subdirectory:", request_body["subdirectory"])
    
    os.makedirs( request_body["subdirectory"] , exist_ok=True)

    if os.path.exists( request_body["subdirectory"] ):
        print('subdirectory:',  request_body["subdirectory"] )
    else:
        print('subdirectory does not exist:',  request_body["subdirectory"] )


    model = model_fn("model")
    predict_fn(request_body, model)
   

if __name__ == "__main__":
    main()
import requests
import torch
from transformers import pipeline


def transcribe_audio(audio_file):
    pipe = pipeline(
        'automatic-speech-recognition',
        model='openai/whisper-tiny.en',
        chunk_length_s=30
    )

    prediction = pipe(sample, batch_size=8)['text']

    return prediction


def download_file(url, file_path):
    response = requests.get(url)


    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)

        print('File downloaded successfully')

        return file_path
    else:
        print('Failed to download the audio file')




if __name__ == '__main__':
    url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-GPXX04C6EN/Testing%20speech%20to%20text.mp3"
    sample = download_file(url, 'download_audio.mp3')

    prediction = transcribe_audio(sample)

    print(prediction)

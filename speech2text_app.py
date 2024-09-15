import gradio as gr

from simple_speech2text import transcribe_audio


def interface_setup():
    audio_input = gr.Audio(sources='upload', type='filepath')
    output_text = gr.Textbox()

    iface = gr.Interface(
        fn=transcribe_audio,
        inputs=audio_input,
        outputs=output_text,
        title='Audio Transcription App',
        description='Upload the audio file'
    )

    return iface


if __name__ == '__main__':
    iface = interface_setup()
    iface.launch(server_name="0.0.0.0", server_port=7860)

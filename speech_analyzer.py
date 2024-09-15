from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from simple_llm import create_instance
from simple_speech2text import transcribe_audio as transcribe_audio_to_text
from speech2text_app import interface_setup

llm = create_instance()

temp = """
<s><<SYS>>
List the key points with details from the context:
[INST] The context : {context} [/INST]
<</SYS>>
"""

pt = PromptTemplate(
    input_variables=['context'],
    template=temp
)

prompt_to_LLAMA2 = LLMChain(llm=llm, prompt=pt)

def transcribe_audio(audio_file):
    transcription_text = transcribe_audio_to_text(audio_file)

    result = prompt_to_LLAMA2.run(transcription_text)

    return result


if __name__ == '__main__':
    iface = interface_setup()
    iface.launch(server_name="0.0.0.0", server_port=7860)

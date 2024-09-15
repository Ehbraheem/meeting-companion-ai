from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams



def create_instance():
    credentials = { 'url': 'https://us-south.ml.cloud.ibm.com' }
    params = {
        GenParams.MAX_NEW_TOKENS: 800,
        GenParams.TEMPERATURE: 0.1,
    }

    LLAMA2_model = Model(
        model_id='meta-llama/llama-2-70b-chat',
        credentials=credentials,
        params=params,
        project_id='skills-network'
    )

    return WatsonxLLM(LLAMA2_model)




def respond_to_prompt(prompt):
    llm = create_instance()

    return llm(prompt)



if __name__ == '__main__':
    prompt = 'How to read a book effectively?'
    response = respond_to_prompt(prompt)
    print(response)
import gradio as gr
from typing import Dict
from langchain.chat_models.openai import ChatOpenAI
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains.conversation.base import ConversationChain
from typing import Optional, Tuple
from chains.slot_memory import SlotMemory
from chains.prompt import CHAT_PROMPT
from configs.params import ModelParams

model_config = ModelParams()

chain: ConversationChain


def initial_chain():
    # llm = AzureChatOpenAI(name="DDLGAI", model="general-model", max_tokens=1024, max_retries=3,
    #                       azure_endpoint=model_config.azure_endpoint, temperature=model_config.temperature, 
    #                       api_key=model_config.azure_openai_key, api_version=model_config.azure_openai_version)
    # llm = AzureChatOpenAI(name="DDLGAI", model="general-model", azure_endpoint=model_config.azure_endpoint, temperature=model_config.temperature, api_key=model_config.azure_openai_key, api_version=model_config.azure_openai_version)
    # llm = ChatOpenAI(name="DDLGAI", temperature=model_config.temperature, openai_api_key=model_config.openai_api_key)
    llm = GoogleGenerativeAI(name="DDLGAI", model="gemini-pro", google_api_key=model_config.gemini_api_key)
    memory = SlotMemory(llm=llm)
    global chain
    chain = ConversationChain(llm=llm, memory=memory, prompt=CHAT_PROMPT)


def clear_session():
    initial_chain()
    return [], []


def slot_format(slot_dict: Dict):
    # result = f"name: {slot_dict['name']}\norigin: {slot_dict['origin']}\ndestination: {slot_dict['destination']}\ndeparture_time: {slot_dict['departure_time']}\n"
    result = f"""
    name: {slot_dict['name']}
    vehicle_condition: {slot_dict['vehicle_condition']}
    previous_accidents: {slot_dict['previous_accidents']}
    vehicle_model: {slot_dict['vehicle_model']}
    mileage: {slot_dict['mileage']}
    age: {slot_dict['age']}
    previous_purchased_insurance: {slot_dict['previous_purchased_insurance']}
    insurance_plan: {slot_dict['insurance_plan']}
    """
    return result


def predict(command, history: Optional[Tuple[str, str]]):
    history = history or []
    response = chain.invoke(input=command)['response']
    # print(response)
    current_slot = chain.memory.current_slots
    history.append((command, response))
    return history, history, '', slot_format(current_slot)


if __name__ == "__main__":
    title = """
    # Slot Filling Demo
    """
    css = """
    footer.svelte-k95dp1 {
        visibility: hidden;
    }
    """
    with gr.Blocks(css=css) as demo:
        gr.Markdown(title)

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot()
                user_input = gr.Textbox(show_label=False, placeholder="Input...", container=False)
                # voice_input = gr.Audio(sources=["microphone"], type="filepath",format="mp3")
                # voice_input.upload()
                with gr.Row():
                    submitBtn = gr.Button("🚀Submit", variant="primary")
                    emptyBtn = gr.Button("🧹Clear History")
            slot_show = gr.Textbox(label="current_slot", lines=20, interactive=False, scale=1)

        initial_chain()
        state = gr.State([])

        submitBtn.click(fn=predict, inputs=[user_input, state], outputs=[chatbot, state, user_input, slot_show])
        emptyBtn.click(fn=clear_session, inputs=[], outputs=[chatbot, state])

    demo.queue().launch(share=False, inbrowser=True, server_name="0.0.0.0", server_port=8000)

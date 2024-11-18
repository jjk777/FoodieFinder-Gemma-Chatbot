import gradio as gr
import os
import keras
import keras_nlp
from langchain.memory import ConversationBufferMemory

# Set the backend to TensorFlow
os.environ["KERAS_BACKEND"] = "tensorflow"

# Load the fine-tuned Gemma model
gemma_finetuned = keras_nlp.models.CausalLM.from_preset("hf://Jeevan18/FoodieFinderV2")

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Function to generate response with conversation memory
def launch(input_text):
    # Retrieve the previous conversation history
    conversation_history = "\n".join([msg.content for msg in memory.chat_memory.messages])
    #print("The conversation history is ", conversation_history)

    # Create the prompt with conversation history and new question
    template = f"{conversation_history}\nQuestion:\n{input_text}\n\nAnswer:\n"

    #print("printing the template for debugging",template)
    
    # Generate the response using the model
    try:
        out = gemma_finetuned.generate(template, max_length=800)
        answer_index = out.index("Answer:") + len("Answer:") + 1
        answer = out[answer_index:].strip()
        
        # Add user question and model answer to memory
        memory.chat_memory.add_user_message(input_text)
        memory.chat_memory.add_ai_message(answer)
        #print("The input text is ", input_text)
        #print("The answer is ", answer)
    except Exception as e:
        answer = f"An error occurred: {str(e)}"
    
    return answer

# Function to clear memory
def clear_memory():
    memory.chat_memory.clear()
    return "Conversation history cleared!"

# Set up Gradio interface with input, output, and clear memory button
with gr.Blocks() as iface:
    gr.Markdown("# FoodieFinder Chatbot 🍣 \n### Your Guide to the Best Dining in Japan: Ask me a question about food in Japna:)")
    gr.Markdown("""
### About This App
FoodieFinder chatbot is an AI tool for foodies designed to help you explore the culinary wonders of Japan. Whether you're looking for the best sushi in Tokyo, cozy ramen spots in Kansai region, or the finest Hokkaido seafood, FoodieFinder is here to simplify your dining and travel decisions. 

You can ask questions about:
- Regional cuisines and specialties. E.g. recommended dishes to try in Hokkaido, Kansai, Chugoku, Okinawa etc
- Dining recommendations based on your atmosphere preferences. E.g. restaurants known for unique dishes, and calm and relaxing environment. Characteristics to choose from include authentic food, great service, cleanliness, calm and relaxing environmen, unique dishes, comfortable atmosphere, chef craftsmanship, friendly and attentive staff, warm and inviting atmosphere, beautiful ambiance, delicious food, romantic setting, generous portions
- Famous food categories such as sushi, ramen, tempura, sake etc

### How to Use:
1. Type a question in the **'Ask a question'** box below.
2. Receive detailed recommendations or insights about Japanese food and dining.
3. If starting a new topic or switching conversations, remember to **clear the memory** using the **'Delete Memory'** button. [This step is required to be done with the memory integrated version of chatbot. Currently, the memory feature is turned off.]

### Example Questions:
- "What restaurants are known for friendly and attentive staff, and chef craftsmanship?"
- "Can you share some information about [name of one of the restaurants from output]". E.g. "Can you share some information about Sushi Karasu Umeda Ten."
- "What is the average rating?"

The above three questions form a conversation. However, due to memory integration issues, I have disabled memory which means the chatbot will no longer remember the previous conversation. It is advised to ask questions as follows:
- "What restaurants are known for friendly and attentive staff, and chef craftsmanship?"
- "Can you share some information about Sushi Karasu Umeda Ten."
- "What is the average rating of Sushi Karasu Umeda Ten?"
- "What is the full address of Sushi Karasu Umeda Ten?"

### Limitations:
Note: Currently, the memory feature is turned off.
With the memory integrated version of the chatbot, when switching topics, you might need to clear the memory to ensure accurate responses. This is due to integration issues between LangChain and the Gemma model.

### Other prompts:
- "What are the seven regions of Japan known for their distinct culinary traditions?"
- "Recommend a place to try customizable ramen."
- "What are the different types of ramen available in Japan?"
- "Can you list some famous ramen shops for Tonkotsu ramen and their unique characteristics?"
- "Which ramen shops are famous for Miso ramen?"
- "What are some tips for visiting ramen shops in Japan?"
- "What are the main differences in ramen styles across Japan?"
- "What restaurants are known for friendly and attentive staff, and beautiful ambiance?"
- "What restaurants are known for authentic food, and great service?"
- "What restaurants are known for great service, and romantic setting?"

    """)
    
    with gr.Row():
        input_box = gr.Textbox(label="Ask a question")
        output_box = gr.Textbox(label="Chatbot Response")
    
    submit_button = gr.Button("Submit")
    clear_button = gr.Button("Delete Memory")
    
    # Link buttons to their respective functions
    submit_button.click(fn=launch, inputs=input_box, outputs=output_box)
    clear_button.click(fn=clear_memory, inputs=None, outputs=output_box)

# Launch the Gradio interface
iface.launch()


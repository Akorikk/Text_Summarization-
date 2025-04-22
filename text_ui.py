from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import gradio as gr

# Load the model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("model_directory")
tokenizer = AutoTokenizer.from_pretrained("model_directory")

# Define the summarization function
def summarize(text, min_length, max_length):
    inputs = tokenizer.encode(text, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs, min_length=min_length, max_length=max_length)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Define custom CSS for additional styling
custom_css = """
    .gradio-container {
        background: linear-gradient(to right, #6a11cb, #2575fc);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .gradio-input, .gradio-output {
        border-radius: 10px;
        border: 1px solid #ccc;
        padding: 15px;
        font-family: 'Arial', sans-serif;
    }
    .gradio-slider {
        background-color: #f0f0f0;
        border-radius: 10px;
    }
    .gradio-button {
        background-color: #2575fc;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .gradio-button:hover {
        background-color: #6a11cb;
    }
"""

# Create the Gradio interface with the Soft theme and custom CSS
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# Text Summarization")
    gr.Markdown("Enter your text below and adjust the sliders to customize the summary length.")
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Input Text", lines=10)
            min_length = gr.Slider(minimum=10, maximum=100, step=1, label="Min Length", value=30)
            max_length = gr.Slider(minimum=10, maximum=200, step=1, label="Max Length", value=100)
            submit_button = gr.Button("Generate Summary")
        with gr.Column():
            output_text = gr.Textbox(label="Summary", lines=10)
    
    submit_button.click(fn=summarize, inputs=[input_text, min_length, max_length], outputs=output_text)

# Launch the interface
demo.launch()
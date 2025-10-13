import gradio as gr
from pathlib import Path
import sys
import os

# Get the path to the 'src' directory by moving up two levels
# current_dir -> src/app
# up_one -> src
# up_two -> Project Root
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the 'src' directory to the system path
sys.path.insert(0, src_dir) # Use insert(0, ...) to ensure it's checked first

from pipeline.rag_llm_pipeline import rag_llm_pipeline
from frontendHelpers import load_user_files_to_documents
from generator.parseOutput import parse_llm_json

# 🔹 Define the Gradio interface function
def run_rag(prompt: str, files_list: list):
    
    # --- STEP 1: Process Documents ---
    loaded_docs = None
    if files_list:
        # Load the files provided by the user into LangChain Document objects
        loaded_docs = load_user_files_to_documents(files_list)
        print(f"Successfully loaded {len(loaded_docs)} document chunks from user files.")
    
    # --- STEP 2: Call RAG Pipeline ---
    # The pipeline is called with `documents=loaded_docs`.
    # If loaded_docs is None, the pipeline will fall back to the default/old logic.
    result = rag_llm_pipeline(query=prompt, documents=loaded_docs)

    # --- STEP 3: Parse LLM Output ---
    try:
        structured = parse_llm_json(result.get("result", ""))
        title = structured.get("title", "No title found.")
        article = structured.get("article", "No article found.")
        formatted_output = f"📰 **{title}**\n\n{article}"
    except Exception as e:
        formatted_output = f"❌ Error parsing response:\n{e}\n\nRaw Output:\n{result}"

    return formatted_output

# 🔹 Gradio UI layout
with gr.Blocks(title="RAG Newspaper article generator") as demo:
    gr.Markdown("## 🧩 RAG Newspaper article generator")
    gr.Markdown("Upload one or more files and enter your prompt for getting a newspaper article below:")

    with gr.Row():
        # Change label to reflect multiple files
        file_input = gr.File(label="Upload Files", file_count="multiple") 
        prompt_input = gr.Textbox(label="Your Prompt", placeholder="Give me the topic for the article...")

    with gr.Row():
        submit_btn = gr.Button("Generate Response")

    output_box = gr.Textbox(label="RAG Response", lines=10)

    # Pass the files_list from the gr.File component
    submit_btn.click(fn=run_rag, inputs=[prompt_input, file_input], outputs=output_box) 

# 🔹 Launch app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", # Allows access from external machines
        server_port=7860,      # Default Gradio port
        share=False             # Creates a public link via Gradio's servers
    )
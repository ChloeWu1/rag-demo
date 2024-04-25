import gradio as gr

def greet(name):
    return "Hello " + name + "!"

interface = gr.Interface(fn=greet, inputs="text", outputs="text")
interface.queue().launch(share=True)

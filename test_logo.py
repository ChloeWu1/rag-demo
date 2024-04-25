import gradio as gr

with gr.Blocks(
    theme=gr.themes.Soft(),
    css=".disclaimer {font-variant-caps: all-small-caps;}",
) as demo:
    # Insert logo image in the top left corner
    demo.top_left = gr.Image(value="logo_trans.png", width=100, height=100, interactive=False, show_share_button=False, show_label=False, show_download_button=False, css={"background": "none"})
    gr.Markdown("""<h1><center>QA over Document</center></h1>""")
    gr.Markdown(f"""<center>Powered by OpenVINO </center>""")

demo.launch(share=True)
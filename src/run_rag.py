# Select model for inference
import os
import time
import webbrowser
import threading
from threading import Thread
from pathlib import Path
from typing import List

import torch
import gradio as gr
import openvino as ov

from optimum.intel.openvino import (
    OVModelForSequenceClassification,
    OVModelForFeatureExtraction,
)
from transformers import (
    AutoTokenizer,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from langchain_community.embeddings import OpenVINOBgeEmbeddings
from langchain_community.document_compressors.openvino_rerank import OpenVINOReranker
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
)
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from llm_config import (
    SUPPORTED_EMBEDDING_MODELS,
    SUPPORTED_RERANK_MODELS,
    SUPPORTED_LLM_MODELS,
)

model_languages = list(SUPPORTED_LLM_MODELS)

model_language = model_languages[1]
print("model language is:", model_language)

# Convert LLM model
llm_model_ids = [model_id for model_id, model_config in SUPPORTED_LLM_MODELS[model_language].items() if model_config.get("rag_prompt_template")]
llm_model_id = llm_model_ids[0]
llm_model_configuration = SUPPORTED_LLM_MODELS[model_language][llm_model_id]
print(f"Selected LLM model {llm_model_id}")

src_dir = os.path.abspath(os.path.dirname(__file__))
model_dir = Path(os.path.join(os.path.dirname(src_dir), "model")) / llm_model_id

fp16_model_dir = model_dir / "FP16"
int8_model_dir = model_dir / "INT8_compressed_weights"
int4_model_dir = model_dir / "INT4_compressed_weights"

fp16_weights = fp16_model_dir / "openvino_model.bin"
int8_weights = int8_model_dir / "openvino_model.bin"
int4_weights = int4_model_dir / "openvino_model.bin"

if fp16_weights.exists():
    print(f"Size of FP16 model is {fp16_weights.stat().st_size / 1024 / 1024:.2f} MB")
for precision, compressed_weights in zip([8, 4], [int8_weights, int4_weights]):
    if compressed_weights.exists():
        print(f"Size of model with INT{precision} compressed weights is {compressed_weights.stat().st_size / 1024 / 1024:.2f} MB")
    if compressed_weights.exists() and fp16_weights.exists():
        print(f"Compression rate for INT{precision} model: {fp16_weights.stat().st_size / compressed_weights.stat().st_size:.3f}")

# Convert embedding model
embedding_model_ids = list(SUPPORTED_EMBEDDING_MODELS[model_language])
embedding_model_id = embedding_model_ids[0]
embedding_model_configuration = SUPPORTED_EMBEDDING_MODELS[model_language][embedding_model_id]
print(f"Selected embedding model {embedding_model_id}")

embedding_model_dir = os.path.join(os.path.dirname(src_dir), "model", embedding_model_id)
if not embedding_model_dir:
    ov_model = OVModelForFeatureExtraction.from_pretrained(embedding_model_configuration["model_id"], compile=False, export=True)
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_configuration["model_id"])
    ov_model.half()
    ov_model.save_pretrained(embedding_model_dir)
    tokenizer.save_pretrained(embedding_model_dir)

# Convert rerank model
rerank_model_ids = list(SUPPORTED_RERANK_MODELS)
rerank_model_id = rerank_model_ids[0]
rerank_model_configuration = SUPPORTED_RERANK_MODELS[rerank_model_id]
print(f"Selected rerank model {rerank_model_id}")

rerank_model_dir = os.path.join(os.path.dirname(src_dir), "model", rerank_model_id)
if not rerank_model_dir:
    ov_model = OVModelForSequenceClassification.from_pretrained(rerank_model_configuration["model_id"], compile=False, export=True)
    tokenizer = AutoTokenizer.from_pretrained(rerank_model_configuration["model_id"])
    ov_model.half()
    ov_model.save_pretrained(rerank_model_dir)
    tokenizer.save_pretrained(rerank_model_dir)


# Select device
core = ov.Core()


def get_user_input(prompt, timeout=10):
    print(prompt)
    input_string = None
    def wait_for_input():
        nonlocal input_string
        input_string = input()
    user_thread = threading.Thread(target=wait_for_input)
    user_thread.start()
    user_thread.join(timeout=timeout)
    if user_thread.is_alive():
        print("No input received within the time limit. Defaulting to 'GPU'.")
        return "GPU"
    return input_string

def select_device():
    options = {"1": "GPU", "2": "CPU", "3": "AUTO"}
    user_input = get_user_input("Please enter the device for model loading (GPU=1, CPU=2, AUTO=3):")
    return options.get(user_input, "GPU")

embedding_device = select_device()
print(f"Embedding model will be loaded to {embedding_device} device for text embedding.")

rerank_device = select_device()
print(f"Rerank model will be loaded to {rerank_device} device for text reranking.")

llm_device = select_device()
print(f"LLM model will be loaded to {llm_device} device for response generation.")


embedding_model_kwargs = {"device": embedding_device}
encode_kwargs = {
    "mean_pooling": embedding_model_configuration["mean_pooling"],
    "normalize_embeddings": embedding_model_configuration["normalize_embeddings"],
}

embedding = OpenVINOBgeEmbeddings(
    model_name_or_path=embedding_model_dir,
    model_kwargs=embedding_model_kwargs,
    encode_kwargs=encode_kwargs,
)

text = "This is a test document."
embedding_result = embedding.embed_query(text)
embedding_result[:3]


rerank_model_kwargs = {"device": rerank_device}
rerank_top_n = 2

reranker = OpenVINOReranker(
    model_name_or_path=rerank_model_dir,
    model_kwargs=rerank_model_kwargs,
    top_n=rerank_top_n,
)
available_models = []
if int4_model_dir.exists():
    available_models.append("INT4")
if int8_model_dir.exists():
    available_models.append("INT8")
if fp16_model_dir.exists():
    available_models.append("FP16")

model_to_run = available_models[0]
print("The compressed weights of model is", model_to_run)

if model_to_run == "INT4":
    model_dir = int4_model_dir
elif model_to_run == "INT8":
    model_dir = int8_model_dir
else:
    model_dir = fp16_model_dir
print(f"Loading model from {model_dir}")

cache_dir = os.path.join(os.path.dirname(os.getcwd()), "model", "cache_dir")
os.makedirs(cache_dir, exist_ok=True)
ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": cache_dir}

# On a GPU device a model is executed in FP16 precision. For red-pajama-3b-chat model there known accuracy
# issues caused by this, which we avoid by setting precision hint to "f32".
if llm_model_id == "red-pajama-3b-chat" and "GPU" in core.available_devices and llm_device in ["GPU", "AUTO"]:
    ov_config["INFERENCE_PRECISION_HINT"] = "f32"

llm = HuggingFacePipeline.from_model_id(
    model_id=str(model_dir),
    task="text-generation",
    backend="openvino",
    model_kwargs={
        "device": llm_device,
        "ov_config": ov_config,
        "trust_remote_code": True,
    },
    pipeline_kwargs={"max_new_tokens": 2},
)

llm.invoke("2 + 2 =")


# Run QA over Document
class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", "\n", text)
            text = text.replace("\n\n", "")
        sent_sep_pattern = re.compile('([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)
        return sent_list


TEXT_SPLITERS = {
    "Character": CharacterTextSplitter,
    "RecursiveCharacter": RecursiveCharacterTextSplitter,
    "Markdown": MarkdownTextSplitter,
    "Chinese": ChineseTextSplitter,
}


LOADERS = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

chinese_examples = [
    ["英特尔®酷睿™ Ultra处理器可以降低多少功耗？"],
    ["相比英特尔之前的移动处理器产品，英特尔®酷睿™ Ultra处理器的AI推理性能提升了多少？"],
    ["英特尔博锐® Enterprise系统提供哪些功能？"],
]

english_examples = [
    ["How much power consumption can Intel® Core™ Ultra Processors help save?"],
    ["Compared to Intel’s previous mobile processor, what is the advantage of Intel® Core™ Ultra Processors for Artificial Intelligence?"],
    ["What can Intel vPro® Enterprise systems offer?"],
]

if model_language == "English":
    text_example_path = "inputs/text_example_en.pdf"
else:
    text_example_path = "inputs/text_example_cn.pdf"

examples = chinese_examples if (model_language == "Chinese") else english_examples

stop_tokens = llm_model_configuration.get("stop_tokens")


class StopOnTokens(StoppingCriteria):
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


if stop_tokens is not None:
    if isinstance(stop_tokens[0], str):
        stop_tokens = llm.pipeline.tokenizer.convert_tokens_to_ids(stop_tokens)

    stop_tokens = [StopOnTokens(stop_tokens)]


def load_single_document(file_path: str) -> List[Document]:
    """
    helper for loading a single document

    Params:
      file_path: document path
    Returns:
      documents loaded

    """
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADERS:
        loader_class, loader_args = LOADERS[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"File does not exist '{ext}'")


def default_partial_text_processor(partial_text: str, new_text: str):
    """
    helper for updating partially generated answer, used by default

    Params:
      partial_text: text buffer for storing previosly generated text
      new_text: text update for the current step
    Returns:
      updated text string

    """
    partial_text += new_text
    return partial_text


text_processor = llm_model_configuration.get("partial_text_processor", default_partial_text_processor)


def create_vectordb(
    docs,
    spliter_name,
    chunk_size,
    chunk_overlap,
    vector_search_top_k,
    vector_search_top_n,
    run_rerank,
):
    """
    Initialize a vector database

    Params:
      doc: orignal documents provided by user
      chunk_size:  size of a single sentence chunk
      chunk_overlap: overlap size between 2 chunks
      vector_search_top_k: Vector search top k

    """
    documents = []
    for doc in docs:
        documents.extend(load_single_document(doc.name))

    text_splitter = TEXT_SPLITERS[spliter_name](chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    texts = text_splitter.split_documents(documents)

    global db
    db = Chroma.from_documents(texts, embedding)

    global retriever
    retriever = db.as_retriever(search_kwargs={"k": vector_search_top_k})
    if run_rerank:
        reranker.top_n = vector_search_top_n
        retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=retriever)
    prompt = PromptTemplate.from_template(llm_model_configuration["rag_prompt_template"])

    global combine_docs_chain
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    
    global rag_chain
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return "Vector database is Ready"


def update_retriever(vector_search_top_k, vector_rerank_top_n, run_rerank, search_method):
    """
    Update retriever

    Params:
      vector_search_top_k: size of searching results
      vector_rerank_top_n:  size of rerank results
      run_rerank: whether run rerank step
      search_method: search method used by vector store

    """
    global retriever
    global db
    global rag_chain
    global combine_docs_chain

    retriever = db.as_retriever(search_kwargs={"k": vector_search_top_k}, search_type=search_method)
    if run_rerank:
        retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=retriever)
        reranker.top_n = vector_rerank_top_n
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)


def user(message, history):
    """
    callback function for updating user messages in interface on submit button click

    Params:
      message: current message
      history: conversation history
    Returns:
      None
    """
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]


def bot(history, temperature, top_p, top_k, repetition_penalty, hide_full_prompt):
    """
    callback function for running chatbot on submit button click

    Params:
      history: conversation history
      temperature:  parameter for control the level of creativity in AI-generated text.
                    By adjusting the `temperature`, you can influence the AI model's probability distribution, making the text more focused or diverse.
      top_p: parameter for control the range of tokens considered by the AI model based on their cumulative probability.
      top_k: parameter for control the range of tokens considered by the AI model based on their cumulative probability, selecting number of tokens with highest probability.
      repetition_penalty: parameter for penalizing tokens based on how frequently they occur in the text.
      hide_full_prompt: whether to show searching results in promopt.

    """
    streamer = TextIteratorStreamer(
        llm.pipeline.tokenizer,
        timeout=60.0,
        skip_prompt=hide_full_prompt,
        skip_special_tokens=True,
    )
    llm.pipeline._forward_params = dict(
        max_new_tokens=512,
        temperature=temperature,
        do_sample=temperature > 0.0,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        streamer=streamer,
    )
    if stop_tokens is not None:
        llm.pipeline._forward_params["stopping_criteria"] = StoppingCriteriaList(stop_tokens)

    t1 = Thread(target=rag_chain.invoke, args=({"input": history[-1][0]},))
    t1.start()

    # Initialize an empty string to store the generated text
    partial_text = ""
    for new_text in streamer:
        partial_text = text_processor(partial_text, new_text)
        history[-1][1] = partial_text
        yield history


def request_cancel():
    llm.pipeline.model.request.cancel()


with gr.Blocks(
    theme=gr.themes.Soft(),
    css=".disclaimer {font-variant-caps: all-small-caps;}",
) as demo:
    gr.Markdown("""<h1><center>QA over Document</center></h1>""")
    gr.Markdown(f"""<center>Powered by OpenVINO and {llm_model_id} </center>""")
    gr.Markdown(f"""<center>MeteorLake Optimized by OpenVINO </center>""")

    # Insert logo image in the top left corner
    demo.top_left = gr.Image(value="utils/logo_trans.png", width=100, height=100, interactive=False, show_share_button=False, show_label=False, show_download_button=False)

    with gr.Row():
        with gr.Column(scale=1):
            docs = gr.File(
                label="Step 1: Load text files",
                value=[text_example_path],
                file_count="multiple",
                file_types=[
                    ".csv",
                    ".doc",
                    ".docx",
                    ".enex",
                    ".epub",
                    ".html",
                    ".md",
                    ".odt",
                    ".pdf",
                    ".ppt",
                    ".pptx",
                    ".txt",
                ],
            )
            load_docs = gr.Button("Step 2: Build Vector Store")
            db_argument = gr.Accordion("Vector Store Configuration", open=False)
            with db_argument:
                spliter = gr.Dropdown(
                    ["Character", "RecursiveCharacter", "Markdown", "Chinese"],
                    value="RecursiveCharacter",
                    label="Text Spliter",
                    info="Method used to splite the documents",
                    multiselect=False,
                )

                chunk_size = gr.Slider(
                    label="Chunk size",
                    value=700,
                    minimum=100,
                    maximum=2000,
                    step=50,
                    interactive=True,
                    info="Size of sentence chunk",
                )

                chunk_overlap = gr.Slider(
                    label="Chunk overlap",
                    value=100,
                    minimum=0,
                    maximum=400,
                    step=10,
                    interactive=True,
                    info=("Overlap between 2 chunks"),
                )

            langchain_status = gr.Textbox(
                label="Vector Store Status",
                value="Vector Store is Not ready",
                interactive=False,
            )
            with gr.Accordion("Generation Configuration", open=False):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            temperature = gr.Slider(
                                label="Temperature",
                                value=0.1,
                                minimum=0.0,
                                maximum=1.0,
                                step=0.1,
                                interactive=True,
                                info="Higher values produce more diverse outputs",
                            )
                    with gr.Column():
                        with gr.Row():
                            top_p = gr.Slider(
                                label="Top-p (nucleus sampling)",
                                value=1.0,
                                minimum=0.0,
                                maximum=1,
                                step=0.01,
                                interactive=True,
                                info=(
                                    "Sample from the smallest possible set of tokens whose cumulative probability "
                                    "exceeds top_p. Set to 1 to disable and sample from all tokens."
                                ),
                            )
                    with gr.Column():
                        with gr.Row():
                            top_k = gr.Slider(
                                label="Top-k",
                                value=50,
                                minimum=0.0,
                                maximum=200,
                                step=1,
                                interactive=True,
                                info="Sample from a shortlist of top-k tokens — 0 to disable and sample from all tokens.",
                            )
                    with gr.Column():
                        with gr.Row():
                            repetition_penalty = gr.Slider(
                                label="Repetition Penalty",
                                value=1.1,
                                minimum=1.0,
                                maximum=2.0,
                                step=0.1,
                                interactive=True,
                                info="Penalize repetition — 1.0 to disable.",
                            )
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                height=600,
                label="Step 3: Input Query",
            )
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        msg = gr.Textbox(
                            label="QA Message Box",
                            placeholder="Chat Message Box",
                            show_label=False,
                            container=False,
                        )
                with gr.Column():
                    with gr.Row():
                        submit = gr.Button("Submit")
                        stop = gr.Button("Stop")
                        clear = gr.Button("Clear")
            gr.Examples(examples, inputs=msg, label="Click on any example and press the 'Submit' button")
            retriever_argument = gr.Accordion("Retriever Configuration", open=True)
            with retriever_argument:
                with gr.Row():
                    with gr.Row():
                        do_rerank = gr.Checkbox(
                            value=True,
                            label="Rerank searching result",
                            interactive=True,
                        )
                        hide_context = gr.Checkbox(
                            value=True,
                            label="Hide searching result in prompt",
                            interactive=True,
                        )
                    with gr.Row():
                        search_method = gr.Dropdown(
                            ["similarity", "mmr"],
                            value="similarity",
                            label="Searching Method",
                            info="Method used to search vector store",
                            multiselect=False,
                            interactive=True,
                        )
                    with gr.Row():
                        vector_rerank_top_n = gr.Slider(
                            1,
                            10,
                            value=2,
                            step=1,
                            label="Rerank top n",
                            info="Number of rerank results",
                            interactive=True,
                        )
                    with gr.Row():
                        vector_search_top_k = gr.Slider(
                            1,
                            50,
                            value=10,
                            step=1,
                            label="Search top k",
                            info="Number of searching results, must >= Rerank top n",
                            interactive=True,
                        )
    load_docs.click(
        create_vectordb,
        inputs=[
            docs,
            spliter,
            chunk_size,
            chunk_overlap,
            vector_search_top_k,
            vector_rerank_top_n,
            do_rerank,
        ],
        outputs=[langchain_status],
        queue=False,
    )
    submit_event = msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot,
        [chatbot, temperature, top_p, top_k, repetition_penalty, hide_context],
        chatbot,
        queue=True,
    )
    submit_click_event = submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot,
        [chatbot, temperature, top_p, top_k, repetition_penalty, hide_context],
        chatbot,
        queue=True,
    )
    stop.click(
        fn=request_cancel,
        inputs=None,
        outputs=None,
        cancels=[submit_event, submit_click_event],
        queue=False,
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    vector_search_top_k.release(
        update_retriever,
        [vector_search_top_k, vector_rerank_top_n, do_rerank, search_method],
    )
    vector_rerank_top_n.release(
        update_retriever,
        [vector_search_top_k, vector_rerank_top_n, do_rerank, search_method],
    )
    do_rerank.change(
        update_retriever,
        [vector_search_top_k, vector_rerank_top_n, do_rerank, search_method],
    )
    search_method.change(
        update_retriever,
        [vector_search_top_k, vector_rerank_top_n, do_rerank, search_method],
    )


demo.queue()
# if you are launching remotely, specify server_name and server_port
#  demo.launch(server_name='your server name', server_port='server port in int')
# if you have any issue to launch on your platform, you can pass share=True to launch method:
# demo.launch(share=True)
# it creates a publicly shareable link for the interface. Read more in the docs: https://gradio.app/docs/


def open_browser():
    time.sleep(1)
    webbrowser.open_new(demo.local_url)


threading.Thread(target=open_browser).start()

demo.launch()

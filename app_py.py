import streamlit as st
import os
import tempfile

st.set_page_config(page_title="MedReport AI", page_icon="🩺", layout="wide")
st.title("🩺 Medical Report Intelligence Assistant")
st.divider()

with st.sidebar:
    st.header("⚙️ Setup")
    uploaded_file = st.file_uploader("Upload Medical Report (PDF)", type=["pdf"])
    process_btn = st.button("⚡ Analyse Report", use_container_width=True)
    st.warning("⚠️ For informational purposes only. Always consult a doctor.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if process_btn:
    if not uploaded_file:
        st.sidebar.error("Please upload a PDF.")
    else:
        with st.spinner("Reading report..."):
            from langchain_community.document_loaders import PyPDFLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import PromptTemplate
            from langchain_core.runnables import RunnablePassthrough
            from langchain_core.output_parsers import StrOutputParser

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            documents = PyPDFLoader(tmp_path).load()
            docs = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)
            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
            vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

            llm = ChatOpenAI(
                model="llama-3.3-70b-versatile",
                api_key=st.secrets["GROQ_API_KEY"],
                base_url="https://api.groq.com/openai/v1"
            )

            prompt = PromptTemplate(
                template="""You are a compassionate medical report assistant. When given a medical report, you MUST automatically provide:

1. 📋 HEALTH SUMMARY - Overall health status in simple words
2. ⚠️ ABNORMAL VALUES - List every abnormal value with what it means
3. ✅ NORMAL VALUES - Brief mention of what is normal
4. 💊 WHAT IT MEANS - Plain English explanation of the condition
5. 🥗 LIFESTYLE RECOMMENDATIONS - Diet, exercise, sleep tips based on the results
6. 👨‍⚕️ DOCTOR ADVICE - When to see a doctor and what to ask

Always be specific to the actual values in the report. Never say "consult a doctor" without first giving your own detailed analysis.

Context: {context}
Question: {question}
Answer:""",
                input_variables=["context", "question"]
            )

            def fmt(docs): return "\n\n".join(d.page_content for d in docs)

            st.session_state.qa_chain = (
                {"context": retriever | fmt, "question": RunnablePassthrough()}
                | prompt | llm | StrOutputParser()
            )
            st.session_state.messages = []
            os.unlink(tmp_path)

        st.sidebar.success(f"✅ Done! {len(documents)} pages loaded.")

        with st.spinner("Generating full report analysis..."):
            auto_q = "Analyse my full medical report and give me a complete health summary, all abnormal values, what they mean, and lifestyle recommendations."
            auto_answer = st.session_state.qa_chain.invoke(auto_q)
            st.session_state.messages.append({"role": "assistant", "content": auto_answer})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if st.session_state.qa_chain:
    user_input = st.chat_input("Ask about your report...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.qa_chain.invoke(user_input)
            st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("👈 Upload your PDF and click Analyse Report to start.")

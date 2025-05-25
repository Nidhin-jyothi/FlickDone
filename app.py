import streamlit as st
import os
import io
import tempfile
from pathlib import Path
import pandas as pd
from datetime import datetime
import json

# Core RAG components
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# Voice processing
import speech_recognition as sr
import pyttsx3
import whisper
from gtts import gTTS
import pygame
import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.playback import play

# Environment setup
from dotenv import load_dotenv
load_dotenv()

class MultilingualVoiceRAG:
    def __init__(self):
        """Initialize the multilingual voice RAG system."""
        self.setup_voice_components()
        self.setup_rag_components()
        self.conversation_history = []
        
    def setup_voice_components(self):
        """Initialize voice processing components."""
        try:
            # Initialize Whisper for STT
            self.whisper_model = whisper.load_model("base")
            
            # Initialize TTS engine
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            
            # Initialize speech recognizer
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Initialize pygame for audio playback
            pygame.mixer.init()
            
            # Supported languages
            self.supported_languages = {
                'en': 'English',
                'es': 'Spanish', 
                'fr': 'French',
                'de': 'German',
                'it': 'Italian',
                'pt': 'Portuguese',
                'ru': 'Russian',
                'ja': 'Japanese',
                'ko': 'Korean',
                'zh': 'Chinese'
            }
            
            st.success("✅ Voice components initialized successfully")
            
        except Exception as e:
            st.error(f"❌ Error initializing voice components: {str(e)}")
            
    def setup_rag_components(self):
        """Initialize RAG system components with Gemini 1.5 Pro."""
        try:
            # Get API key from environment or Streamlit secrets
            google_api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
            
            if not google_api_key:
                st.error("❌ Google API Key not found. Please set GOOGLE_API_KEY in environment variables or Streamlit secrets.")
                st.info("💡 Get your API key from: https://makersuite.google.com/app/apikey")
                return
            
            # Configure Gemini API
            genai.configure(api_key=google_api_key)
            
            # Initialize Gemini 1.5 Pro model
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest",
                google_api_key=google_api_key,
                temperature=0.3,
                max_tokens=2048,
                convert_system_message_to_human=True
            )
            
            # Local embedding model
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            # Vector store path
            self.vector_store_path = "voice_rag_index"
            self.vector_store = None
            
            st.success("✅ RAG components with gemini-1.5-flash-latest initialized successfully")
            
        except Exception as e:
            st.error(f"❌ Error initializing RAG components: {str(e)}")
            st.info("💡 Make sure you have a valid Google API key and internet connection.")
    
    def process_documents(self, uploaded_files):
        """Process uploaded documents and create vector store."""
        try:
            all_text = ""
            
            for uploaded_file in uploaded_files:
                if uploaded_file.type == "application/pdf":
                    pdf_reader = PdfReader(uploaded_file)
                    for page in pdf_reader.pages:
                        all_text += page.extract_text()
                elif uploaded_file.type == "text/plain":
                    all_text += str(uploaded_file.read(), "utf-8")
            
            if not all_text.strip():
                st.warning("⚠️ No text content found in uploaded files")
                return False
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            text_chunks = text_splitter.split_text(all_text)
            
            # Create vector store
            self.vector_store = FAISS.from_texts(
                text_chunks,
                embedding=self.embeddings
            )
            
            # Save vector store
            self.vector_store.save_local(self.vector_store_path)
            
            st.success(f"✅ Processed {len(text_chunks)} text chunks from documents")
            return True
            
        except Exception as e:
            st.error(f"❌ Error processing documents: {str(e)}")
            return False
    
    def load_vector_store(self):
        """Load existing vector store."""
        try:
            if os.path.exists(self.vector_store_path):
                self.vector_store = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                return True
            return False
        except Exception as e:
            st.error(f"❌ Error loading vector store: {str(e)}")
            return False
    
    def transcribe_audio(self, audio_file, language=None):
        """Transcribe audio using Whisper."""
        try:
            # Load audio with whisper
            if language:
                result = self.whisper_model.transcribe(
                    audio_file,
                    language=language
                )
            else:
                result = self.whisper_model.transcribe(audio_file)
            
            detected_language = result.get('language', 'en')
            transcribed_text = result['text'].strip()
            
            return transcribed_text, detected_language
            
        except Exception as e:
            st.error(f"❌ Error transcribing audio: {str(e)}")
            return "", "en"
    
    def record_audio(self, duration=5):
        """Record audio from microphone."""
        try:
            with self.microphone as source:
                st.info("🎤 Listening... Please speak now!")
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source)
                # Record audio
                audio = self.recognizer.listen(source, timeout=duration)
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio.get_wav_data())
                return tmp_file.name
                
        except sr.WaitTimeoutError:
            st.warning("⚠️ No speech detected within the time limit")
            return None
        except Exception as e:
            st.error(f"❌ Error recording audio: {str(e)}")
            return None
    
    def generate_response(self, query, language="en"):
        """Generate response using RAG with Gemini 1.5 Pro."""
        try:
            if not self.vector_store:
                return "Please upload and process documents first."
            
            if not hasattr(self, 'llm') or self.llm is None:
                return "Gemini model not initialized. Please check your API key."
            
            # Retrieve relevant documents
            docs = self.vector_store.similarity_search(query, k=3)
            
            # Combine document content
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create multilingual prompt optimized for Gemini
            language_name = self.supported_languages.get(language, "English")
            
            prompt_template = """You are a helpful AI assistant designed specifically for visually impaired users. Your role is to provide clear, concise, and accessible answers based on the provided context.

IMPORTANT INSTRUCTIONS:
- Answer in {language} language
- Be conversational and empathetic
- Structure your response clearly for audio consumption
- If information is not in the context, politely say so
- Provide step-by-step explanations when appropriate
- Use simple, clear language

CONTEXT:
{context}

USER QUESTION: {question}

Please provide a helpful response in {language}:"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question", "language"]
            )
            
            # Create QA chain with Gemini
            chain = load_qa_chain(
                self.llm,
                chain_type="stuff",
                prompt=prompt
            )
            
            # Generate response
            response = chain({
                "input_documents": docs,
                "question": query,
                "language": language_name
            }, return_only_outputs=True)
            
            return response["output_text"]
            
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while processing your request. Please try again."
    
    def generate_direct_response(self, query, language="en"):
        """Generate response directly using Gemini without RAG (fallback)."""
        try:
            if not hasattr(self, 'llm') or self.llm is None:
                return "Gemini model not initialized. Please check your API key."
            
            language_name = self.supported_languages.get(language, "English")
            
            prompt = f"""You are a helpful AI assistant for visually impaired users. 
            Please answer the following question in {language_name} language.
            Be clear, concise, and empathetic in your response.
            
            Question: {query}
            
            Answer:"""
            
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            st.error(f"❌ Error generating direct response: {str(e)}")
            return "I apologize, but I encountered an error while processing your request."
    
    def text_to_speech(self, text, language="en", use_gtts=True):
        """Convert text to speech."""
        try:
            if use_gtts and language in ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh']:
                # Use gTTS for better multilingual support
                tts = gTTS(text=text, lang=language, slow=False)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                    tts.save(tmp_file.name)
                    return tmp_file.name
            else:
                # Use pyttsx3 as fallback
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    self.tts_engine.save_to_file(text, tmp_file.name)
                    self.tts_engine.runAndWait()
                    return tmp_file.name
                    
        except Exception as e:
            st.error(f"❌ Error in text-to-speech: {str(e)}")
            return None
    
    def play_audio(self, audio_file):
        """Play audio file."""
        try:
            if audio_file.endswith('.mp3'):
                # Convert MP3 to WAV for pygame
                audio = AudioSegment.from_mp3(audio_file)
                wav_file = audio_file.replace('.mp3', '.wav')
                audio.export(wav_file, format="wav")
                audio_file = wav_file
            
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
                
        except Exception as e:
            st.error(f"❌ Error playing audio: {str(e)}")
    
    def save_conversation(self, user_input, response, language):
        """Save conversation to history."""
        self.conversation_history.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'language': language,
            'user_input': user_input,
            'response': response
        })
    
    def export_conversation_history(self):
        """Export conversation history to CSV."""
        if self.conversation_history:
            df = pd.DataFrame(self.conversation_history)
            return df.to_csv(index=False)
        return ""

def main():
    st.set_page_config(
        page_title="Multilingual Voice RAG Assistant with Gemini",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 Multilingual Voice Chat RAG System")
    st.markdown("*AI Assistant powered by Google Gemini 1.5 Pro for Accessible Content Search and Interaction*")
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = MultilingualVoiceRAG()
    
    rag_system = st.session_state.rag_system
    
    # API Key input section
    if not os.getenv("GOOGLE_API_KEY") and "GOOGLE_API_KEY" not in st.secrets:
        st.warning("🔑 Google API Key Required")
        api_key = st.text_input(
            "Enter your Google API Key:",
            type="password",
            help="Get your API key from https://makersuite.google.com/app/apikey"
        )
        
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            st.rerun()
        else:
            st.info("Please enter your Google API Key to use gemini-1.5-flash-latest")
            st.stop()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model info
        st.info("🤖 **Model:** gemini-1.5-flash-latest")
        
        # Language selection
        selected_language = st.selectbox(
            "Select Language",
            options=list(rag_system.supported_languages.keys()),
            format_func=lambda x: f"{rag_system.supported_languages[x]} ({x})",
            index=0
        )
        
        # Document upload
        st.header("📁 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload Documents",
            accept_multiple_files=True,
            type=['pdf', 'txt'],
            help="Upload PDF or text files to create knowledge base"
        )
        
        if uploaded_files and st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                if rag_system.process_documents(uploaded_files):
                    st.success("Documents processed successfully!")
        
        # Load existing vector store
        if st.button("Load Existing Knowledge Base"):
            if rag_system.load_vector_store():
                st.success("Knowledge base loaded!")
            else:
                st.warning("No existing knowledge base found")
        
        # Direct chat mode
        st.header("💬 Chat Mode")
        use_rag = st.toggle("Use RAG (requires documents)", value=True)
        
        # Conversation history
        st.header("📝 Conversation History")
        if rag_system.conversation_history:
            for i, conv in enumerate(rag_system.conversation_history[-5:]):
                with st.expander(f"Conv {i+1} ({conv['language']})"):
                    st.write(f"**User:** {conv['user_input']}")
                    st.write(f"**Assistant:** {conv['response']}")
        
        if st.button("Export History"):
            if rag_system.conversation_history:
                csv_data = rag_system.export_conversation_history()
                st.download_button(
                    "Download CSV",
                    csv_data,
                    "conversation_history.csv",
                    "text/csv"
                )
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎙️ Voice Interaction")
        
        # Voice input section
        if st.button("🎤 Record Voice Input", key="record_btn"):
            if use_rag and not rag_system.vector_store:
                st.warning("⚠️ Please upload and process documents first, or disable RAG mode!")
            else:
                audio_file = rag_system.record_audio(duration=10)
                if audio_file:
                    st.audio(audio_file)
                    
                    with st.spinner("Transcribing audio..."):
                        transcribed_text, detected_lang = rag_system.transcribe_audio(
                            audio_file, selected_language
                        )
                    
                    if transcribed_text:
                        st.success(f"**Transcribed ({detected_lang}):** {transcribed_text}")
                        
                        with st.spinner("Generating response with Gemini..."):
                            if use_rag:
                                response = rag_system.generate_response(
                                    transcribed_text, detected_lang
                                )
                            else:
                                response = rag_system.generate_direct_response(
                                    transcribed_text, detected_lang
                                )
                        
                        st.write("**Assistant Response:**")
                        st.write(response)
                        
                        # Generate audio response
                        with st.spinner("Generating speech..."):
                            audio_response = rag_system.text_to_speech(
                                response, detected_lang
                            )
                        
                        if audio_response:
                            st.audio(audio_response)
                            
                            if st.button("🔊 Play Response"):
                                rag_system.play_audio(audio_response)
                        
                        # Save conversation
                        rag_system.save_conversation(
                            transcribed_text, response, detected_lang
                        )
                    
                    # Clean up temporary file
                    if os.path.exists(audio_file):
                        os.unlink(audio_file)
        
        # Text input alternative
        st.header("⌨️ Text Input (Alternative)")
        user_query = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="Type your question here..."
        )
        
        if st.button("Submit Text Query") and user_query:
            if use_rag and not rag_system.vector_store:
                st.warning("⚠️ Please upload and process documents first, or disable RAG mode!")
            else:
                with st.spinner("Generating response with Gemini..."):
                    if use_rag:
                        response = rag_system.generate_response(
                            user_query, selected_language
                        )
                    else:
                        response = rag_system.generate_direct_response(
                            user_query, selected_language
                        )
                
                st.write("**Response:**")
                st.write(response)
                
                # Generate audio response
                with st.spinner("Generating speech..."):
                    audio_response = rag_system.text_to_speech(
                        response, selected_language
                    )
                
                if audio_response:
                    st.audio(audio_response)
                
                # Save conversation
                rag_system.save_conversation(
                    user_query, response, selected_language
                )
    
    with col2:
        st.header("ℹ️ System Status")
        
        # System status indicators
        if hasattr(rag_system, 'llm') and rag_system.llm:
            st.success("✅ Gemini 1.5 Pro: Ready")
        else:
            st.error("❌ Gemini 1.5 Pro: Not available")
        
        if rag_system.vector_store:
            st.success("✅ Knowledge Base: Ready")
        else:
            st.warning("⚠️ Knowledge Base: Not loaded")
        
        if rag_system.whisper_model:
            st.success("✅ Speech Recognition: Ready")
        else:
            st.error("❌ Speech Recognition: Not available")
        
        if rag_system.tts_engine:
            st.success("✅ Text-to-Speech: Ready")
        else:
            st.error("❌ Text-to-Speech: Not available")
        
        # Usage instructions
        st.header("📋 Instructions")
        st.markdown("""
        1. **API Key**: Enter your Google API key (required)
        2. **Upload Documents**: Add PDF or text files for RAG
        3. **Process Documents**: Click to index uploaded files
        4. **Select Language**: Choose your preferred language
        5. **Voice Input**: Click record button and speak
        6. **Listen**: Get audio responses from Gemini
        7. **Chat Mode**: Toggle RAG on/off for different use cases
        """)
        
        # Supported languages display
        st.header("🌍 Supported Languages")
        for code, name in rag_system.supported_languages.items():
            st.text(f"{name} ({code})")
        
        # Model information
        st.header("🤖 Model Information")
        st.markdown("""
        **Gemini 1.5 Pro Features:**
        - Advanced reasoning capabilities
        - Multilingual support
        - Large context window (1M tokens)
        - Fast response times
        - High-quality text generation
        """)

if __name__ == "__main__":
    main()
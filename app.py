import gradio as gr
import whisper
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

class AudioSummarizer:
    def __init__(self):
        self.whisper_model = None
        self.summarizer = None
        self.tokenizer = None
        self.load_models()
    
    def load_models(self):
        """Load Whisper and summarization models"""
        try:
            # Load Whisper model (using base for balance of speed/accuracy)
            print("Loading Whisper model...")
            self.whisper_model = whisper.load_model("base")
            
            # Load summarization model (BART-large-cnn for quality summaries)
            print("Loading summarization model...")
            model_name = "google/pegasus-large"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.summarizer = pipeline(
                "summarization",
                model=model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def transcribe_audio(self, audio_file):
        """Transcribe audio using Whisper"""
        if audio_file is None:
            return "No audio file provided"
        
        try:
            # Whisper expects the audio file path
            result = self.whisper_model.transcribe(audio_file)
            return result["text"].strip()
        except Exception as e:
            return f"Error transcribing audio: {str(e)}"
    
    def calculate_summary_params(self, text, summary_level):
        """Calculate summary parameters based on slider value"""
        text_length = len(text.split())
        
        # Summary level: 1 (most detailed) to 10 (most concise)
        # Calculate min/max length based on original text and slider
        base_ratio = 0.3  # Base ratio for most detailed summary
        ratio_reduction = (summary_level - 1) * 0.1 # Reduce ratio as level increases
        final_ratio = max(0.05, base_ratio - ratio_reduction)  # Minimum 5% of original
        
        max_length = max(50, int(text_length * final_ratio))
        min_length = max(20, int(max_length * 0.3))
        
        return min_length, max_length
    
    def summarize_text(self, text, summary_level):
        """Summarize text with specified level of detail"""
        if not text or len(text.strip()) < 50:
            return "Text too short to summarize effectively"
        
        try:
            # Calculate summary parameters
            min_length, max_length = self.calculate_summary_params(text, summary_level)
            
            # Handle long texts by chunking if necessary
            max_input_length = 1024  # BART's typical max input
            
            if len(text.split()) > max_input_length:
                # Simple chunking strategy
                words = text.split()
                chunks = [' '.join(words[i:i+max_input_length]) 
                         for i in range(0, len(words), max_input_length)]
                
                summaries = []
                for chunk in chunks:
                    chunk_min = max(10, min_length // len(chunks))
                    chunk_max = max(30, max_length // len(chunks))
                    
                    summary = self.summarizer(
                        chunk,
                        max_length=chunk_max,
                        min_length=chunk_min,
                        do_sample=False
                    )[0]['summary_text']
                    summaries.append(summary)
                
                # Combine and re-summarize if needed
                combined = ' '.join(summaries)
                if len(combined.split()) > max_length:
                    final_summary = self.summarizer(
                        combined,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False
                    )[0]['summary_text']
                    return final_summary
                else:
                    return combined
            else:
                # Direct summarization for shorter texts
                summary = self.summarizer(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )[0]['summary_text']
                return summary
                
        except Exception as e:
            return f"Error summarizing text: {str(e)}"
    
    def process_audio(self, audio_file, summary_level, progress=gr.Progress()):
        """Main processing function"""
        if audio_file is None:
            return "", "", "Please upload an audio file"
        
        progress(0.1, desc="Starting transcription...")
        
        # Transcribe audio
        progress(0.3, desc="Transcribing audio...")
        transcription = self.transcribe_audio(audio_file)
        
        if transcription.startswith("Error") or transcription == "No audio file provided":
            return transcription, "", transcription
        
        progress(0.7, desc="Generating summary...")
        
        # Summarize transcription
        summary = self.summarize_text(transcription, summary_level)
        
        progress(1.0, desc="Complete!")
        
        # Prepare stats
        original_words = len(transcription.split())
        summary_words = len(summary.split()) if not summary.startswith("Error") else 0
        compression_ratio = f"{(1 - summary_words/original_words)*100:.1f}%" if original_words > 0 else "N/A"
        
        stats = f"""
        📊 **Processing Stats:**
        - Original: {original_words} words
        - Summary: {summary_words} words  
        - Compression: {compression_ratio}
        - Summary Level: {summary_level}/10
        """
        
        return transcription, summary, stats

# Initialize the processor
processor = AudioSummarizer()

# Custom CSS for beautiful UI
custom_css = """
#main-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

.header {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 30px;
    border-radius: 15px;
    margin-bottom: 30px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

.input-section {
    background: #f8f9fa;
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 20px;
    border: 1px solid #e9ecef;
}

.output-section {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    border: 1px solid #e9ecef;
}

.slider-container {
    background: white;
    padding: 20px;
    border-radius: 10px;
    margin: 15px 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.stats-box {
    background: #e8f5e8;
    border-left: 4px solid #28a745;
    padding: 15px;
    margin: 15px 0;
    border-radius: 5px;
}

#process-btn {
    background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
    border: none;
    color: white;
    font-size: 16px;
    font-weight: bold;
    padding: 12px 30px;
    border-radius: 25px;
    box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
    transition: all 0.3s ease;
}

#process-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4);
}

.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
"""

# Create the Gradio interface
def create_interface():
    with gr.Blocks(css=custom_css, title="Audio Summarizer Pro") as interface:
        
        # Header
        gr.HTML("""
        <div class="header">
            <h1>🎵 Audio Summarizer Pro</h1>
            <p>Transform your audio into intelligent summaries with AI-powered transcription and customizable summarization</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<div class="input-section">')
                gr.Markdown("### 🎤 Upload Audio")
                audio_input = gr.Audio(
                    label="Select Audio File", 
                    type="filepath",
                    elem_id="audio-input"
                )
                
                gr.HTML('<div class="slider-container">')
                gr.Markdown("### 🎯 Summarization Level")
                summary_level = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Summary Detail Level",
                    info="1 = Most Detailed | 10 = Most Concise",
                    elem_id="summary-slider"
                )
                gr.HTML('</div>')
                
                process_btn = gr.Button(
                    "🚀 Process Audio", 
                    variant="primary",
                    elem_id="process-btn",
                    size="lg"
                )
                gr.HTML('</div>')
            
            with gr.Column(scale=2):
                gr.HTML('<div class="output-section">')
                gr.Markdown("### 📝 Results")
                
                with gr.Tabs():
                    with gr.TabItem("📄 Transcription"):
                        transcription_output = gr.Textbox(
                            label="Full Transcription",
                            lines=8,
                            placeholder="Your audio transcription will appear here...",
                            show_copy_button=True
                        )
                    
                    with gr.TabItem("✨ Summary"):
                        summary_output = gr.Textbox(
                            label="AI Summary",
                            lines=6,
                            placeholder="Your intelligent summary will appear here...",
                            show_copy_button=True
                        )
                    
                    with gr.TabItem("📊 Stats"):
                        stats_output = gr.Markdown(
                            value="Processing statistics will appear here after analysis..."
                        )
                
                gr.HTML('</div>')
        
        # Add examples
        gr.HTML("""
        <div style="margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <h3>💡 Pro Tips:</h3>
            <ul>
                <li><strong>Audio Quality:</strong> Clear audio = better transcription results</li>
                <li><strong>Summary Levels:</strong> Start with level 5, then adjust based on your needs</li>
                <li><strong>File Formats:</strong> Supports MP3, WAV, M4A, and more</li>
                <li><strong>Length:</strong> Works best with audio under 30 minutes</li>
            </ul>
        </div>
        """)
        
        # Event handlers
        process_btn.click(
            fn=processor.process_audio,
            inputs=[audio_input, summary_level],
            outputs=[transcription_output, summary_output, stats_output],
            show_progress=True
        )
        
        # Add footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 40px; padding: 20px; color: #6c757d;">
            <p>Built with ❤️ using Whisper AI & BART | Powered by Gradio</p>
        </div>
        """)
    
    return interface

# Launch the app
if __name__ == "__main__":
    try:
        print("🚀 Starting Audio Summarizer Pro...")
        interface = create_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,  # Creates public link for sharing
            show_error=True,
            favicon_path=None,
            app_kwargs={"docs_url": "/docs"}
        )
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        print("Make sure you have all required dependencies installed:")
        print("pip install gradio whisper torch transformers librosa")
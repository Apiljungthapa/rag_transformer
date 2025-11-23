import io
import os
import random
import PyPDF2
import torch
from transformer import CustomTokenizer, MLMTaskHead, TransformerEncoder
from typing import Dict, List, Tuple, Optional
import numpy as np

class PDFProcessor:
    """
    Handles PDF text extraction and processing
    """

    @staticmethod
    def load_pdf_from_path(file_path: str) -> bytes:
        """Load PDF file from path and return as bytes"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if not file_path.lower().endswith('.pdf'):
            raise ValueError("File must be a PDF (.pdf extension)")

        with open(file_path, 'rb') as file:
            return file.read()

    @staticmethod
    def extract_text_from_pdf(pdf_content: bytes) -> str:
        """Extract text from PDF bytes"""
        try:
            pdf_file = io.BytesIO(pdf_content)
            reader = PyPDF2.PdfReader(pdf_file)

            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"

            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting PDF text: {str(e)}")

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for processing"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

class TalkToPDFSystem:
    """
    Main system that combines all components for PDF interaction
    """

    def __init__(self, d_model: int = 768, num_heads: int = 12, num_layers: int = 6):
        self.tokenizer = CustomTokenizer(vocab_size=10000)
        self.pdf_processor = PDFProcessor()

        self.encoder = None
        self.mlm_head = None

        self.pdf_text = ""
        self.pdf_chunks = []
        self.pdf_embeddings = None

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

    def load_pdf_from_path(self, pdf_path: str):
        """Load PDF from file path"""
        print(f"Loading PDF from: {pdf_path}")
        pdf_content = self.pdf_processor.load_pdf_from_path(pdf_path)
        self.load_pdf(pdf_content)

    def load_pdf(self, pdf_content: bytes):
        """Load and process PDF content"""
        print("Extracting text from PDF...")
        self.pdf_text = self.pdf_processor.extract_text_from_pdf(pdf_content)

        if not self.pdf_text.strip():
            raise ValueError("No text could be extracted from the PDF")

        self.pdf_chunks = self.pdf_processor.chunk_text(self.pdf_text)

        print("Building vocabulary from PDF content...")
        self.tokenizer.build_vocab([self.pdf_text])

        print("Initializing transformer model...")
        vocab_size = len(self.tokenizer.vocab)
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers
        )
        self.mlm_head = MLMTaskHead(self.d_model, vocab_size)

        print("Processing PDF content...")
        self._process_pdf_content()

        print(f"PDF loaded successfully! Extracted {len(self.pdf_text.split())} words.")
        print(f"Created {len(self.pdf_chunks)} chunks for processing.")
        print(f"Vocabulary size: {len(self.tokenizer.vocab)} tokens.")

    def _process_pdf_content(self):
        """Process PDF content through the model to get embeddings"""
        embeddings = []
        print(f"Processing {len(self.pdf_chunks)} chunks...")

        for i, chunk in enumerate(self.pdf_chunks):

            token_ids = self.tokenizer.encode(chunk, max_length=512)
            input_ids = torch.tensor([token_ids])

            with torch.no_grad():
                hidden_states, _ = self.encoder(input_ids)

                chunk_embedding = hidden_states[0, 0, :].numpy()
                embeddings.append(chunk_embedding)

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(self.pdf_chunks)} chunks")

        self.pdf_embeddings = np.array(embeddings)

    def create_mlm_training_data(self, text: str, mask_prob: float = 0.15) -> Tuple[List[int], List[int], List[int]]:
        """
        Create training data for Masked Language Modeling
        Returns: (input_ids, labels, mask_positions)
        """
        token_ids = self.tokenizer.encode(text, max_length=512)
        original_ids = token_ids.copy()

        labels = [-100] * len(token_ids)
        mask_positions = []

        for i, token_id in enumerate(token_ids):
            if (token_id not in [self.tokenizer.special_tokens['[CLS]'],
                                self.tokenizer.special_tokens['[SEP]'],
                                self.tokenizer.special_tokens['[PAD]']] and
                random.random() < mask_prob):

                mask_positions.append(i)
                labels[i] = original_ids[i]

                if random.random() < 0.8:
                    token_ids[i] = self.tokenizer.special_tokens['[MASK]']

                elif random.random() < 0.5:
                    token_ids[i] = random.randint(5, len(self.tokenizer.vocab) - 1)

        return token_ids, labels, mask_positions

    def demonstrate_mlm(self, text: str):
        """
        Demonstrate Masked Language Modeling on given text
        """
        print(f"\n=== MLM Demonstration ===")
        print(f"Original text: {text}")

        masked_ids, labels, mask_positions = self.create_mlm_training_data(text)

        print(f"Masked text: {self.tokenizer.decode(masked_ids)}")
        print(f"Mask positions: {mask_positions}")

        input_ids = torch.tensor([masked_ids])
        hidden_states, attention_weights = self.encoder(input_ids)

        mlm_logits = self.mlm_head(hidden_states)

        print("\nPredictions for masked tokens:")
        for pos in mask_positions:
            if pos < len(masked_ids):
                predicted_id = torch.argmax(mlm_logits[0, pos]).item()
                predicted_token = self.tokenizer.inverse_vocab.get(predicted_id, '[UNK]')
                original_token = self.tokenizer.inverse_vocab.get(labels[pos], '[UNK]')
                print(f"Position {pos}: Predicted '{predicted_token}' | Original '{original_token}'")

        return mlm_logits, attention_weights

    def find_relevant_chunks(self, query: str, top_k: int = 3) -> List[str]:
        """
        Find most relevant PDF chunks for a query using simple similarity
        """
        if self.pdf_embeddings is None:
            return self.pdf_chunks[:top_k]

        query_ids = self.tokenizer.encode(query, max_length=512)
        input_ids = torch.tensor([query_ids])

        with torch.no_grad():
            hidden_states, _ = self.encoder(input_ids)
            query_embedding = hidden_states[0, 0, :].numpy()

        similarities = []
        for chunk_embedding in self.pdf_embeddings:
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding))
            similarities.append(similarity)

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        relevant_chunks = [self.pdf_chunks[i] for i in top_indices]



        return relevant_chunks

    def answer_question(self, question: str) -> str:
        """
        Answer a question based on PDF content
        """
        print(f"\nQuestion: {question}")

        relevant_chunks = self.find_relevant_chunks(question, top_k=3)

        print("Finding relevant content in PDF...")

        context = " ".join(relevant_chunks)

        sentences = context.split('.')
        relevant_sentences = []

        question_words = set(question.lower().split())
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())

            overlap = len(question_words.intersection(sentence_words))
            if overlap > 1:
                relevant_sentences.append(sentence.strip())

        if relevant_sentences:
            answer = ". ".join(relevant_sentences[:3])
        else:
            answer = "I couldn't find specific information about your question in the PDF."

        print(f"\nAnswer: {answer}")
        return answer

    def show_pdf_summary(self):
        """Show summary of loaded PDF"""
        if not self.pdf_text:
            print("No PDF loaded yet.")
            return

        words = self.pdf_text.split()
        sentences = self.pdf_text.split('.')

        print(f"\n=== PDF Summary ===")
        print(f"Total characters: {len(self.pdf_text)}")
        print(f"Total words: {len(words)}")
        print(f"Total sentences: {len(sentences)}")
        print(f"Total chunks: {len(self.pdf_chunks)}")
        print(f"Vocabulary size: {len(self.tokenizer.vocab)}")

        lines = self.pdf_text.split('\n')[:5]
        print(f"\nFirst few lines:")
        for i, line in enumerate(lines, 1):
            if line.strip():
                print(f"{i}: {line.strip()[:100]}...")

    def interactive_chat(self):
        """
        Start interactive chat session with the PDF
        """
        print("\n" + "="*60)
        print("🤖 Welcome to Talk-to-PDF System!")
        print("="*60)
        print("Commands:")
        print("  • Ask any question about the PDF content")
        print("  • Type 'mlm <text>' to demonstrate MLM on custom text")
        print("  • Type 'summary' to see PDF statistics")
        print("  • Type 'help' to see this menu again")
        print("  • Type 'quit' to exit")
        print("="*60)

        while True:
            try:
                user_input = input("\n🗣️  You: ").strip()

                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  • Ask any question about the PDF content")
                    print("  • Type 'mlm <text>' to demonstrate MLM")
                    print("  • Type 'summary' to see PDF statistics")
                    print("  • Type 'quit' to exit")
                elif user_input.lower() == 'summary':
                    self.show_pdf_summary()
                elif user_input.lower().startswith('mlm '):

                    text = user_input[4:]
                    if text.strip():
                        self.demonstrate_mlm(text)
                    else:
                        print("Please provide text after 'mlm'. Example: mlm The weather is nice today")
                else:

                    if user_input.strip():
                        self.answer_question(user_input)
                    else:
                        print("Please enter a question or command.")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
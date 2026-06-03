import io
from pdf2image import convert_from_path
from PIL import Image
import google.generativeai as genai
import os
import tempfile


class LocalMedicalPDFIngestor:
    def __init__(
        self,
        pdf_path: str,
        model_name: str = "gemini-2.5-flash"
    ):
        self.pdf_path = pdf_path
        self.model_name = model_name

        genai.configure(
            api_key=os.getenv("GEMINI_API_KEY")
        )

        self.model = genai.GenerativeModel(
            self.model_name
        )

    def _get_image_bytes(self, image: Image) -> bytes:
        """Converts a PIL Image to raw bytes."""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return buffered.getvalue()

    def transcribe_page(
        self,
        image_bytes: bytes,
        page_num: int
    ) -> str:
        """Passes image bytes to Gemini."""

        print(f"[PAGE {page_num}] Sending request to Gemini...")

        prompt = """
        You are an expert medical transcriptionist.

        Read the following medical document page and extract ALL text exactly as written.

        - If there is handwriting, do your best to transcribe it.
        - If there is a table (like ICU charts, vitals, or labs), format it cleanly as a Markdown table.
        - Ignore generic anatomical diagrams (like the human body outlines for bed sores) UNLESS there are handwritten notes pointing to specific areas.
        - If there are notes, just transcribe the notes.
        - Do not add any conversational filler.

        Just return the extracted text.
        """

        try:

            response = self.model.generate_content(
                [
                    prompt,
                    {
                        "mime_type": "image/jpeg",
                        "data": image_bytes,
                    },
                ]
            )

            print(f"[PAGE {page_num}] Gemini response received.")

            return response.text

        except Exception as e:
            print(
                f"Error calling Gemini on page {page_num}: {e}"
            )
            return (
                f"[ERROR TRANSCRIBING PAGE {page_num}]"
            )

    def process_pdf(self) -> str:
        """Converts PDF and compiles transcript."""
        import tempfile # Ensure this is imported at the top of your script

        print(f"Converting {self.pdf_path} to images...")
        full_transcript = []

        # Create the temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            print("Starting optimized PDF conversion...")
            pages = convert_from_path(
                self.pdf_path,
                dpi=150,               
                fmt="jpeg",            
                thread_count=4,        
                output_folder=temp_dir, 
                poppler_path=r"C:\Users\Sarth\Downloads\poppler-26.02.0\Library\bin" # Keep your poppler path here
            )
            print(f"Converted {len(pages)} pages.")

            # SUB-SAMPLING: Only process the first 5 pages for rapid development
            testing_pages = pages[:5]
            print(f"Limiting transcription to first {len(testing_pages)} pages for testing...")

            for i, page in enumerate(testing_pages):
                print(f"Transcribing page {i+1}/{len(pages)} using {self.model_name}...")
                
                # 1. Extract the bytes
                img_bytes = self._get_image_bytes(page)
                
                # 2. THE FIX: Explicitly close the image to release the Windows file lock
                page.close()
                
                # 3. Send to Gemini
                page_text = self.transcribe_page(img_bytes, i + 1)
                full_transcript.append(f"--- PAGE {i+1} ---\n{page_text}")

            # Close any remaining pages that we skipped to ensure the folder can delete cleanly
            for page in pages[5:]:
                page.close()

        # The temp_dir will now successfully delete itself here
        print("All pages processed successfully.")
        
        master_raw_text = "\n\n".join(full_transcript)
        
        # Save the transcript to a file so we don't have to run this again!
        with open("patient_2_raw_transcript.txt", "w", encoding="utf-8") as f:
            f.write(master_raw_text)
            print("Saved transcript to patient_2_raw_transcript.txt")

        return master_raw_text


if __name__ == "__main__":
    ingestor = LocalMedicalPDFIngestor(
        pdf_path=r"C:\Users\Sarth\Downloads\patient.pdf",
        model_name="gemini-2.5-flash"
    )

    print("Starting PDF processing...")
    final_raw_text = ingestor.process_pdf()

    print("\nProcessing complete.")
    print(final_raw_text[:1000])

from pdf_process import TalkToPDFSystem
from typing import Dict, List, Tuple, Optional


def main():
    """Main function to run the Talk-to-PDF system"""
    print("="*60)
    print("🚀 Talk-to-PDF System Initialization")
    print("="*60)

    system = TalkToPDFSystem(
        d_model=384,
        num_heads=6,
        num_layers=4
    )


    while True:
        pdf_path = r"C:\Users\apilt\Desktop\interview_prj\391104eng.pdf"

        if pdf_path.lower() == 'quit':
            return

        try:

            system.load_pdf_from_path(pdf_path)
            break
        except Exception as e:
            print(f"❌ Error loading PDF: {e}")
            print("Please try again or type 'quit' to exit.")

    system.show_pdf_summary()

    system.interactive_chat()

if __name__ == "__main__":
    main()
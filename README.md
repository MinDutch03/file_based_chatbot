# Gemini File Chatbot 
https://duc-multi-filechat.streamlit.app/

Gemini File Chatbot is a Streamlit-based application that allows users to chat with a conversational AI model trained on documents. The chatbot extracts information from uploaded files and answers user questions based on the provided context.



## Features

- **PDF, Text, Image Upload:** Users can upload multiple files.
- **Text Extraction:** Extracts text from uploaded files.
- **Conversational AI:** Uses the Gemini conversational AI model to answer user questions.
- **Chat Interface:** Provides a chat interface to interact with the chatbot.

## Getting Started

If you have docker installed, you can run the application using the following command:

- Obtain a Google API key and set it in the `.env` file.

  ```.env
  GOOGLE_API_KEY=your_api_key_here
  ```

```bash
docker-compose up --build
```

Your application will be available at [http://localhost:8501](http://localhost:8501).

### Deploying your application to the cloud

First, build your image, e.g.: `docker build -t myapp .`.
If your cloud uses a different CPU architecture than your development
machine (e.g., you are on a Mac M1 and your cloud provider is amd64),
you'll want to build the image for that platform, e.g.:
`docker build --platform=linux/amd64 -t myapp .`.

Then, push it to your registry, e.g. `docker push myregistry.com/myapp`.

Consult Docker's [getting started](https://docs.docker.com/go/get-started-sharing/)
docs for more detail on building and pushing.

### References

- [Docker&#39;s Python guide](https://docs.docker.com/language/python/)

## Local Development

Follow these instructions to set up and run this project on your local machine.

   **Note:** This project requires Python 3.10 or higher.

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/MinDutch03/pdf_chatbot.git
   ```
2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   pip install pytesseract pillow
   ```
3. **Set up Tesseract:**
   **On Windows:**
   - Download Tesseract here: https://github.com/UB-Mannheim/tesseract/wiki
   - Add Path to the system environment: `/path/to/Tesseract-OCR`

4. **Set up Google API Key:**

   - Obtain a Google API key and set it in the `.env` file.

   ```bash
   GOOGLE_API_KEY=your_api_key_here
   ```
5. **Run the Application:**

   ```bash
   streamlit run main.py
   ```
6. **Upload Files:**

   - Use the sidebar to upload files.
   - Click on "Submit & Process" to extract text and generate embeddings.
7. **Chat Interface:**

   - Chat with the AI in the main interface.

## Project Structure

- `app.py`: Main application script.
- `.env`: file which will contain your environment variable.
- `requirements.txt`: Python packages required for working of the app.
- `README.md`: Project documentation.

## Dependencies

- PyPDF2
- langchain
- Streamlit
- google.generativeai
- dotenv
- tesseract


## Acknowledgments

- [Google Gemini](https://ai.google.com/): For providing the underlying language model.
- [Streamlit](https://streamlit.io/): For the user interface framework.

# PDF Summarization Script

This Python script automates the summarization of multiple PDF files using large language models (LLMs) from OpenAI (GPT-3.5-turbo) or Anthropic (Claude-3.5-Haiku-latest). It processes all PDFs in a specified directory, generates summaries, and saves them to a customizable output file in either text or JSON format. The script is flexible, allowing customization of the model, directory, output file, and format, with summaries saved to a user-specified file for easy access.

## Features

- **Configuration**: Loads environment variables (API keys, default model) from a `.env` file using `python-dotenv`.
- **Command-Line Arguments**: Parses user inputs for model selection (`openai` or `anthropic`), PDF directory, output file name, and output format (`text` or `json`) using argparse.
- **PDF Processing**: Scans the specified directory for `.pdf` files and processes each one using `PyPDFLoader` to extract text.
- **Model Selection**: Choose between OpenAI's `GPT-3.5-turbo` or Anthropic's `Claude-3.5-Haiku-latest` via command-line arguments or environment variables.
- **Summarization**: Uses LangChain's `map_reduce` summarization chain to generate summaries with the chosen LLM, handling errors gracefully.
- **Flexible Output**: Save summaries to a custom file name:
    - **Text Format**: Appends each summary to a `.txt` file with headers and separators.
    - **JSON Format**: Collects summaries in a dictionary and saves them as a `.json` file.
- **Error Handling**: Gracefully handles errors (e.g., invalid PDFs) and includes error messages in the output.
- **Console Feedback**: Prints progress, summaries, and the final output file location.

## Requirements

- Python 3.8+
- Required Python packages (install via `pip`):
```bash
pip install langchain openai anthropic pypdf python-dotenv
```
- API keys for OpenAI and/or Anthropic (depending on the model used).
- PDF files to summarize.

## Setup

**1. Clone the Repository (if applicable):**
```bash
git clone <repository-url>
cd <repository-directory>
```

**2. Install Dependencies:**
```bash
pip install -r requirements.txt
```

Alternatively, install the required packages manually:
```bash
pip install langchain openai anthropic pypdf python-dotenv
```

Packages and Versions:
- `langchain==0.2.16`: For the summarization chain and document loading utilities.
- `openai==1.35.13`: For accessing OpenAI's GPT-3.5-turbo model.
- `anthropic==0.28.0`: For accessing Anthropic's Claude-3.5-Haiku-latest model.
- `pypdf==4.2.0`: For reading and parsing PDF files.
- `python-dotenv==1.0.1`: For loading environment variables from a .env file.

**3. Configure Environment Variables:** Create a `.env` file in the project root with the following content:
```plain
OPENAI_API_KEY="your-openai-api-key"
ANTHROPIC_API_KEY="your-anthropic-api-key"
LLM_MODEL="openai"  # Optional: default model ("openai" or "anthropic")
```

Replace `your-openai-api-key` and `your-anthropic-api-key` with your actual API keys. The `LLM_MODEL` variable is optional and sets the default model if not specified via command-line.

**4. Prepare PDFs:** Place the PDF files you want to summarize in a directory (e.g., the project root or a subfolder like `pdfs`).


## Usage
Run the script using the `python main.py` command with optional arguments to customize its behavior.

**Command-Line Arguments**

- `--model`: Specify the LLM to use (`openai` or `anthropic`). Defaults to `LLM_MODEL` environment variable or `openai`.
- `--directory`: Directory containing PDF files. Defaults to the current directory (`.`).
- `--output`: Output file name (without extension). Defaults to `summaries`.
- `--format`: Output format (`text` or `json`). Defaults to `text`.

**Examples**

**1. Summarize PDFs in the current directory using OpenAI, saving as text:**
```bash
python main.py
```

Output file: `summaries.txt`

**2. Use Anthropic, specify a directory, and save as JSON:**
```bash
python main.py --model anthropic --directory ./pdfs --output my_summaries --format json
```

Output file: `my_summaries.json`

**3. Use OpenAI, custom output file, text format:**
```bash
python main.py --model openai --directory ./pdfs --output results
```

Output file: `results.txt`


## Output

- Console Output: The script prints the model used, each PDF being processed, its summary, and the final output file location.
- Text Format (e.g., `summaries.txt`):
```plain
Summary for document1.pdf:
This document discusses the impact of climate change on coastal ecosystems.
==================================================

Summary for document2.pdf:
Error processing document2.pdf: Invalid PDF format
==================================================

Summary for document3.pdf:
The report covers advancements in renewable energy technologies.
==================================================
```

- JSON Format (e.g., `summaries.json`):
```json
{
  "document1.pdf": "This document discusses the impact of climate change on coastal ecosystems.",
  "document2.pdf": "Error processing document2.pdf: Invalid PDF format",
  "document3.pdf": "The report covers advancements in renewable energy technologies."
}
```

## Notes

- **API Keys**: Ensure the appropriate API key is set in `.env` for the selected model. Missing keys will cause authentication errors.
- **File Overwrite**: The output file is overwritten each time the script runs. To append instead, modify the script by removing the `os.remove(output_file)` line.
- **Error Handling**: If a PDF fails to process (e.g., corrupt file), the error is included in the output, and the script continues with the next file.
- **Extending the Script**:
    - Add more output formats (e.g., CSV) by extending the `save_summary_*` functions.
    - Support additional LLMs by updating the `get_llm` function.
    - Enable parallel processing for faster handling of large PDF collections.

## Troubleshooting

- **ModuleNotFoundError**: Ensure all dependencies are installed (`pip install -r requirements.txt`).
- **Authentication Errors**: Verify that API keys in `.env` are correct and match the selected model.
- **No PDFs Found**: Check that the specified directory contains `.pdf` files and the path is correct.
- **Invalid PDF**: If a PDF fails to process, the error will be included in the output, and other PDFs will still be processed.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or contributions, please open an issue or submit a pull request on the repository.


## Deep Learning with Long Short-Term Memory (LSTM) for Auto-Tagging

This project applies a deep learning approach using LSTM networks to automatically tag text content. It specifically aims to tag Stack Overflow questions.

### Project Structure

- `data/`: Contains datasets and the zipped file.
  - `archive (2).zip`: Zipped dataset file.
  - `unzipped_contents/`: Extracted datasets, including `Answers.csv`, `Questions.csv`, `Tags.csv`, and `database.sqlite`.
- `notebook/`: Jupyter notebooks for experimentation and interactive development.
  - `dllstm.ipynb`: The main Jupyter notebook for the project.
- `source/`: Source code for the project.
  - `dllstm.py`: Python script implementing the LSTM model.
- `weights.best.hdf5`: Model weights saved during the training process.
- `LICENSE`: The license document for the project.
- `README.md`: Documentation and guidelines for using this project.
- `requirements.txt`: List of Python dependencies for the project.

## Installation

Ensure Python 3 and pip are already installed on your system. Then, follow these steps:

1. Clone the repository to your local machine.

2. Navigate to the project directory and install the required dependencies:



# requirements.txt

```txt
pandas
beautifulsoup4
matplotlib
numpy
scikit-learn
tensorflow
```

Run the command below to install the necessary packages:

```bash
pip install -r requirements.txt
```


3. Unzip the dataset:


```bash
unzip data/archive\(2\).zip -d data/unzipped_contents
```

## Usage

Run the Jupyter notebook for a detailed walkthrough:

```bash
jupyter notebook notebook/dllstm.ipynb
```

Alternatively, you can run the Python script directly:

```bash
python source/dllstm.py
```

## Workflow

1. **Load Data and Import Libraries**: Setup and import necessary Python libraries.
2. **Text Cleaning**: Implement a function to clean the text data.
3. **Merge Tags with Questions**: Combine question data with corresponding tags.
4. **Dataset Preparation**: Prepare the data for model training.
5. **Text Representation**: Convert the text to a suitable format for the LSTM model.
6. **Model Building**: Construct and define the LSTM model architecture.
7. **Model Predictions**: Make tag predictions using the trained model.
8. **Model Evaluation**: Assess the model's performance.
9. **Inference**: Use the model to predict tags for new text inputs.

## Contributing

Feel free to fork the repository, make improvements, and submit a pull request.

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for more details.

---


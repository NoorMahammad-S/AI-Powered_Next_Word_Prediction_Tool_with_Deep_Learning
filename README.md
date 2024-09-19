# ```                AI & ML Project                ```

# AI-Powered Next Word Prediction Tool with Deep Learning

A AI-Powered Next Word Prediction tool using a Long Short-Term Memory (LSTM) neural network. The model is trained on the Reuters corpus using TensorFlow and NLTK.

## Dependencies

Make sure you have the following dependencies installed:

- Python 3.x
- NumPy
- TensorFlow
- NLTK

You can install the required Python packages using the following command:

```bash
pip install numpy tensorflow nltk
```

You also need to download the NLTK data:

```bash
python -m nltk.downloader punkt
python -m nltk.downloader reuters
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/NoorMahammad-S/AI-Powered_Next_Word_Prediction_Tool_with_Deep_Learning.git
cd AI-Powered_Next_Word_Prediction_Tool_with_Deep_Learning
```

2. Run the `main.py` script to train the LSTM model:

```bash
python main.py
```

3. After training, you can use the generated model to predict the next word in a sequence. Modify the `seed_text` variable in the script to change the starting point for text generation.

```python
# Test the model
seed_text = "The company"
predicted_text = generate_next_word(seed_text, model, tokenizer, max_sequence_length)
print(predicted_text)
```

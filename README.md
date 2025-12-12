# Arabic Gulf EOU Detection (AraBERT)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![LiveKit](https://img.shields.io/badge/LiveKit-Agent-green)
![AraBERT](https://img.shields.io/badge/Model-AraBERT-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📌 Project Overview

**Islamhitham-Arabic_Gulf_EOU_Detection_AraBERT** is a specialized End-of-Utterance (EOU) detection system designed for Arabic conversational AI, with a specific focus on Gulf/Saudi dialects. 

The project fine-tunes **AraBERT** to accurately detect when a user has finished speaking, enabling more natural and responsive turn-taking in real-time voice applications. It includes a complete pipeline from data preprocessing and augmentation to model training and a live agent implementation using **LiveKit** and **Deepgram**.

## ✨ Features

*   **Arabic-Specific Preprocessing**: Robust normalization handling diacritics (Tashkeel), Alef/Yeh/Teh Marbuta normalization, and text cleaning.
*   **Smart Data Augmentation**: Automatically generates negative samples (incomplete utterances) to train the model to distinguish between pauses and actual sentence endings.
*   **Real-time Integration**: Fully functional LiveKit agent (`livekit_agent.py`) that performs real-time EOU checks on both interim and final speech transcripts.
*   **Deepgram STT**: Optimized for Arabic speech recognition (using `nova-2` model).
*   **Visual Debugging**: Console output includes probability scores and Arabic text reshaping for proper display in terminals.

##  Project Structure

```bash
Islamhitham-Arabic_Gulf_EOU_Detection_AraBERT/
├── dataset/                # Data processing scripts
│   ├── data_collection.py  # Data gathering tools
│   ├── data_preprocessing.py # Main processing & augmentation pipeline
│   ├── visualize_data.py   # Data analysis tools
│   └── processed_data/     # Output for cleaned datasets
├── model/                  # Model training and artifact storage
│   └── final_model/        # Saved trained model
├── livekit_eou_sdk/        # Custom SDK for EOU integration
├── livekit_agent.py        # Main LiveKit agent application
├── demo/                   # Demonstration scripts
└── raw_data/               # Source datasets (Saudi_Test, SADA22)
```

##  Getting Started

### Prerequisites

*   Python 3.8+
*   [LiveKit Cloud](https://livekit.io/) Project (URL and API Key)
*   [Deepgram](https://deepgram.com/) API Key (for STT)


##  Usage

### 1. Data Preprocessing
Prepare your raw datasets (Saudi_Test, SADA22) and run the processing pipeline:

```bash
python dataset/data_preprocessing.py
```
This will:
*   Normalize Arabic text.
*   Generate label maps based on punctuation.
*   Create negative samples (incomplete sentences).
*   Save the processed dataset to `dataset/processed_data`.

### 2. Training the Model
Train the AraBERT model on the processed data:

```bash
python model/train.py
```

### 3. Running the Agent
Start the LiveKit agent to test real-time EOU detection:

```bash
python livekit_agent.py
```
The agent will connect to your LiveKit room, transcribe audio using Deepgram, and print EOU probabilities in real-time.

##  How It Works

1.  **Speech Input**: Audio is streamed to Deepgram via LiveKit.
2.  **Transcription**: Deepgram returns interim and final transcripts.
3.  **EOU Analysis**: The text is fed into the fine-tuned AraBERT model.
    *   **Probability Score**: The model calculates a probability (0.0 - 1.0) that the current text represents a complete sentence.
    *   **Decision Logic**: If `probability > threshold` (e.g., 0.7), the agent signals an automated turn-taking event.
4.  **Response**: The system can then interrupt or wait for the user to finish based on this signal.

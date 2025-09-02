# EquiComm 🎯

EquiComm is an AI-powered conversation assistant designed to bridge the gaps of meeting equity, accessibility, and clarity in diverse settings. Built on a passion for leveraging AI to foster fairness and inclusion, EquiComm harnesses recent advancements in speech, emotion, and language processing.

The project targets groups most affected by communication barriers: people with hearing difficulties, underrepresented team members, and global employees facing linguistic challenges.

## 🎯 Problem Statement

In large, diverse meetings — from corporate boardrooms to government councils — communication often fails to be truly inclusive because of three key gaps:

### Voice Equity Gap 
Dominance by certain groups — whether by age, gender, or department — often goes unmeasured, leaving silent exclusivity to persist.

### Accessibility Gap 
For people with hearing difficulties, transcripts capture words but not meaning — missing tone, sarcasm, and emotional cues. For example, "Well, that was great" might read the same, whether genuine or sarcastic, leading to misunderstandings and emotional exclusion.

### Clarity Gap 
In global or cross-cultural meetings, accents, and local phrases lead to misinterpretations. For example, an Indian saying "passed out of college" (meaning graduated) might be taken by an American as literally losing consciousness.

## 💡 Proposed Solution – EquiComm

An AI-powered conversation assistant that addresses these gaps through:

## 🚀 Features

### 🗣️ Voice Equity Dashboard
Visual analytics show speaking time and participation split by demographics.

**Example**: In any project meeting—whether in a corporation, government, classroom, or global team—participants can instantly see if certain roles, departments, or groups have disproportionately less speaking time, making imbalances visible and actionable.

- **Speaker Demographics**: Gender prediction from voice patterns
- **Participation Analytics**: Speaking time distribution and turn-taking analysis
- **Interactive Dashboard**: Visual representation of meeting dynamics

### 😊 Emotion-Enhanced Transcripts
Real-time transcripts with tone annotations and emojis for emotional clarity.

**Example**: Instead of a plain transcript saying "Oh, that's brilliant!", EquiComm presents "[Sarcastic] Oh, that's brilliant! 😏", ensuring hearing-impaired participants understand the true intent behind words.

- **Audio Emotion Detection**: Real-time emotion classification from speech
- **Text Sentiment Analysis**: Context-aware emotion detection from transcripts
- **Sarcasm Detection**: Advanced model for detecting subtle communication patterns
- **Emoji Annotation**: Enriched transcripts with emotional context

### 🌍 Accent & Vocabulary Neutralization
Real-time audio adaptation (listen in your own accent) and transcript suggestions for uncommon terms.

**Example**: Transcripts show "Graduated" on hover over "Passed out of college" based on audience preference — all while preserving tone and voice style.

- **Accent Neutralization**: Standardize pronunciation variations
- **Vocabulary Normalization**: Translate regional terms to standard equivalents
- **Accessibility Enhancement**: Improve understanding across diverse linguistic backgrounds

## � Impact

EquiComm makes these invisible barriers visible and actionable, transforming meetings in corporations, government councils, educational settings, and global teams. By ensuring every voice is heard and every tone is understood, EquiComm drives real inclusivity, accessibility, and clarity—wherever people meet.

### 🏢 Corporate 
Improves meeting inclusivity and data-driven leadership decisions.

### 🏛️ Government 
Promotes equal representation of diverse demographics — including gender, race, caste, age, and more — in public consultations and policy discussions.

### 🎓 Education 
Makes diverse classrooms more accessible for students with different backgrounds or hearing needs.

### 🌐 Global Teams 
Reduces misinterpretation and improves cultural understanding in real-time collaboration.

## �🏗️ Architecture

EquiComm follows a modular, plugin-based architecture:

```
EquiComm/
├── main.py                 # Main entry point
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── utils/                 # Core utilities
│   ├── transcribe_media.py    # Audio transcription
│   ├── print_helpers.py       # Logging utilities
│   └── ms_teams_api.py        # Teams integration
├── plugins/               # Feature modules
│   ├── voice_equity/          # Voice analytics
│   ├── emotion_transcript/    # Emotion detection
│   └── accent_vocab_neutral/  # Language normalization
└── Weekly Meeting Example.mp3 # Sample audio file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone [repository-url]
   cd EquiComm
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Dependencies optimized for performance using faster_whisper (instead of standard whisper) and includes audio processing libraries for enhanced accuracy.*

3. **Download required models** (automatic on first run):
   - Whisper for transcription
   - Transformers models for emotion detection
   - Gender classification models

## 🚀 Quick Start

1. **Basic Usage**:
   ```bash
   python main.py
   ```
   This will process the included sample audio file and generate:
   - Transcription with speaker diarization
   - Emotion-annotated transcript
   - Voice equity analytics
   - Interactive dashboard

2. **Process Your Own Audio**:
   Replace the `AUDIO_FILE` path in `main.py` with your audio file:
   ```python
   AUDIO_FILE = os.path.abspath("your_meeting.mp3")
   ```

## 📊 Output

EquiComm generates several types of output:

1. **Console Logs**: Real-time processing updates with color-coded information
2. **Annotated Transcripts**: Text with emotional context and speaker identification
3. **Analytics Report**: Participation metrics and equity insights
4. **Visual Dashboard**: Charts showing speaking patterns and demographics

## 🔧 Configuration

Edit `config.py` to customize:
- Model paths and preferences
- API tokens (if using cloud services)
- Processing parameters

## 🔌 Plugin System

Each plugin is self-contained and can be used independently:

- **`voice_equity/`**: Speaker analysis and participation metrics
- **`emotion_transcript/`**: Emotion detection and transcript enhancement
- **`accent_vocab_neutral/`**: Language normalization tools

## 🤝 Integration

EquiComm is designed to integrate with:
- Microsoft Teams (via `ms_teams_api.py`)
- Various audio formats (MP3, WAV, etc.)
- Real-time audio streams
- Custom meeting platforms

## 🎯 Use Cases

- **Meeting Facilitation**: Ensure balanced participation
- **Accessibility**: Support diverse linguistic backgrounds
- **Research**: Analyze communication patterns
- **Training**: Improve meeting dynamics awareness
- **HR Analytics**: Evaluate team collaboration patterns

## 🔍 Performance

The system provides detailed timing information for optimization:
- Model loading times
- Processing times per segment
- Memory usage patterns

## 🚧 Development

To extend EquiComm:

1. Create new plugins in the `plugins/` directory
2. Follow the existing plugin structure
3. Update configuration as needed
4. Add new dependencies to `requirements.txt`

## 📝 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]

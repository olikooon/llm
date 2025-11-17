# FitnessBot

Бот для Telegram, предоставляющий персонализированные советы по фитнесу и питанию с использованием искусственного интеллекта Gemini от Google. Специализируется на диетах и упражнениях, 
предоставляя персонализированные рекомендации на основе профилей пользователей.

## Bot Handle
[@diets_and_calisthenics_lab1_bot](https://t.me/diets_and_calisthenics_lab1_bot)

## Setup Instructions

### 1. Prerequisites

- Python 3.8+
- Telegram Bot Token from [@BotFather](https://t.me/BotFather)
- Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/app/api-keys)

### 2. Installation

1. **Clone the project**:

2. **Create a virtual environment**:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install aiogram python-dotenv google-generativeai
```

4. **Configuration**
    1. Create .env file in the project root:
   ```bash
   BOT_TOKEN=your_telegram_bot_token_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
5. **Run**

```bash
python main.py
```

## Usage

### Commands

- `/start` - Begin setup and create your fitness profile
- `/help` - Get help and usage information  
- `/reset` - Clear your profile and start over

### Profile Setup

The bot will guide you through providing:

- **Weight** (in kg)
- **Height** (in cm)
- **Age** (optional)
- **Fitness Goal**: Weight Loss, Muscle Gain, General Fitness, Strength, Endurance, or Flexibility

### Asking Questions

After setup, you can ask about:

- Diet plans and nutrition
- Calisthenics workouts
- Bodyweight exercises
- Weight management
- Exercise progressions

**Example questions**:

- "What's a good diet plan for me?"
- "Create a calisthenics workout routine"
- "How many calories should I eat?"
- "Beginner bodyweight exercises for strength"
import asyncio
import logging
import os

import google.generativeai as genai
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv('BOT_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiService:
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.5-pro')

    async def get_fitness_advice(self, user_data: dict, question: str = "") -> str:
        try:
            prompt = self._create_prompt(user_data, question)

            await asyncio.sleep(0.5)

            response = self.model.generate_content(prompt)
            return response.text

        except Exception as e:
            return f"Sorry, I encountered an error. Please try again later. Error: {str(e)}"

    @staticmethod
    def _create_prompt(user_data: dict, question: str) -> str:
        base_prompt = f"""
                You are a fitness and nutrition expert. Provide advice ONLY about diets and calisthenics.

                User Profile:
                - Weight: {user_data.get('weight', 'Not provided')} kg
                - Height: {user_data.get('height', 'Not provided')} cm  
                - Age: {user_data.get('age', 'Not provided')} years
                - Goal: {user_data.get('goal', 'general fitness')}

                User Question: {question if question else "General fitness advice"}

                STRICT RULES:
                1. ONLY answer questions related to diets, nutrition, calisthenics, bodyweight exercises
                2. If asked about other topics, politely decline and redirect to fitness topics
                3. Provide practical, safe advice suitable for the user's profile
                4. Focus on bodyweight exercises, no gym equipment unless specifically asked
                5. Include both diet and exercise recommendations when appropriate
                6. Use CLEAN, SIMPLE FORMATTING without any markdown, asterisks, or special characters
                7. Use only plain text with clear section breaks using dashes or simple spacing
                8. Do NOT use #, *, **, or any other markdown symbols
                9. Use clear headings with simple capitalization or underlining with dashes

                Provide a comprehensive response with:
                - Diet recommendations
                - Calisthenics workout plan
                - Safety considerations
                - Progressive overload principles

                FORMATTING REQUIREMENTS:
                - Use plain text only
                - Use uppercase for main section titles
                - Use indentation and simple bullet points with dashes
                - Separate sections with blank lines
                - Never use markdown symbols
                """

        return base_prompt


class UserData(StatesGroup):
    weight = State()
    height = State()
    age = State()
    goal = State()
    question = State()


bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
gemini_service = GeminiService()


def get_goals_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Weight Loss"), KeyboardButton(text="Muscle Gain")],
            [KeyboardButton(text="General Fitness"), KeyboardButton(text="Strength")],
            [KeyboardButton(text="Endurance"), KeyboardButton(text="Flexibility")]
        ],
        resize_keyboard=True,
        one_time_keyboard=True
    )


@dp.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext):
    await message.answer(
        "Welcome to FitnessBot!\n\n"
        "I specialize in diets and calisthenics advice.\n\n"
        "To give you personalized recommendations, I need some information:\n"
        "1. Your weight\n"
        "2. Your height  \n"
        "3. Your age\n"
        "4. Your fitness goal\n\n"
        "Let's start! Please enter your weight in kg:"
    )
    await state.set_state(UserData.weight)


@dp.message(UserData.weight)
async def process_weight(message: Message, state: FSMContext):
    try:
        weight = float(message.text)
        if weight <= 0 or weight > 300:
            await message.answer("Please enter a valid weight between 1-300 kg:")
            return

        await state.update_data(weight=weight)
        await message.answer("Great! Now please enter your height in cm:")
        await state.set_state(UserData.height)

    except ValueError:
        await message.answer("Please enter a valid number for weight (e.g., 70):")


@dp.message(UserData.height)
async def process_height(message: Message, state: FSMContext):
    try:
        height = float(message.text)
        if height <= 0 or height > 250:
            await message.answer("Please enter a valid height between 1-250 cm:")
            return

        await state.update_data(height=height)
        await message.answer(
            "Now please enter your age:",
            reply_markup=ReplyKeyboardMarkup(
                keyboard=[[KeyboardButton(text="Skip")]],
                resize_keyboard=True,
                one_time_keyboard=True
            )
        )
        await state.set_state(UserData.age)

    except ValueError:
        await message.answer("Please enter a valid number for height (e.g., 175):")


@dp.message(UserData.age)
async def process_age(message: Message, state: FSMContext):
    if message.text.lower() != "skip":
        try:
            age = int(message.text)
            if age <= 0 or age > 120:
                await message.answer("Please enter a valid age between 1-120:")
                return
            await state.update_data(age=age)
        except ValueError:
            await message.answer("Please enter a valid number for age (e.g., 25):")
            return

    await message.answer(
        "What is your main fitness goal?",
        reply_markup=get_goals_keyboard()
    )
    await state.set_state(UserData.goal)


@dp.message(UserData.goal)
async def process_goal(message: Message, state: FSMContext):
    valid_goals = ["weight loss", "muscle gain", "general fitness", "strength", "endurance", "flexibility"]

    if message.text.lower() not in [goal.lower() for goal in valid_goals]:
        await message.answer("Please select a goal from the keyboard:")
        return

    await state.update_data(goal=message.text)

    user_data = await state.get_data()

    summary = f"""
✅ Profile Complete!

Your information:
• Weight: {user_data.get('weight')} kg
• Height: {user_data.get('height')} cm
• Age: {user_data.get('age', 'Not specified')}
• Goal: {user_data.get('goal')}

Now you can ask me anything about diets and calisthenics!

Examples:
• "What's a good diet plan for me?"
• "Create a calisthenics workout"
• "How many calories should I eat?"
• "Beginner bodyweight exercises"
    """

    await message.answer(summary, reply_markup=ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="Get Diet Plan"), KeyboardButton(text="Get Workout Plan")]],
        resize_keyboard=True
    ))
    await state.set_state(UserData.question)


@dp.message(UserData.question)
async def process_question(message: Message, state: FSMContext):
    user_data = await state.get_data()

    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")

    if message.text == "Get Diet Plan":
        question = "Create a personalized diet plan with calorie recommendations and meal timing"
    elif message.text == "Get Workout Plan":
        question = "Create a personalized calisthenics workout plan with progressions"
    else:
        question = message.text

    try:
        response = await gemini_service.get_fitness_advice(user_data, question)

        if len(response) > 4000:
            parts = [response[i:i + 4000] for i in range(0, len(response), 4000)]
            for part in parts:
                await message.answer(part)
                await asyncio.sleep(0.5)
        else:
            await message.answer(response)

    except Exception as e:
        logger.error(f"Error: {e}")
        await message.answer("Sorry, I'm having trouble processing your request. Please try again.")


@dp.message(Command("help"))
async def cmd_help(message: Message):
    help_text = """
FitnessBot Help

I provide personalized advice about:
• Diets and nutrition
• Calisthenics and bodyweight exercises
• Weight management
• Exercise progressions

Commands:
/start - Start over and update your profile
/help - Show this help message
/reset - reset your profile

Just send me a message about diets or exercises!
"""
    await message.answer(help_text)


@dp.message(Command("reset"))
async def cmd_reset(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("Your profile has been reset. Use /start to begin again.")


@dp.message(F.text)
async def handle_other_topics(message: Message):
    fitness_keywords = [
        'diet', 'exercise', 'workout', 'calorie', 'weight', 'muscle',
        'nutrition', 'food', 'eat', 'train', 'fitness', 'health',
        'pushup', 'pullup', 'squat', 'calisthenics', 'bodyweight'
    ]

    if any(keyword in message.text.lower() for keyword in fitness_keywords):
        await message.answer(
            "I'd love to help with fitness advice! But first I need your profile.\n"
            "Please use /start to set up your profile with weight, height, age, and goals."
        )
    else:
        await message.answer(
            "I specialize only in diets and calisthenics advice.\n\n"
            "Please ask me about:\n"
            "• Diet plans and nutrition\n"
            "• Bodyweight exercises\n"
            "• Calisthenics workouts\n"
            "• Weight management\n\n"
            "Use /start to begin!"
        )


async def main():
    logger.info("Starting FitnessBot...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

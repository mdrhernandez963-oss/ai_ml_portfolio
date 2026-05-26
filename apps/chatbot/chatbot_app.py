# chatbot_app.py
import os
import streamlit as st
from dotenv import load_dotenv

from apps.chatbot.chatbot_core import load_limited_data, get_limited_response
from apps.chatbot.llm_client import generate_llm_response


def run():
     # ✅ FORCE SCROLL TO TOP
    st.markdown(
        "<script>window.scrollTo(0, 0);</script>",
        unsafe_allow_html=True
    )
    # -------------------------------------------------
    # Page setup (IMPORTANT: do NOT call set_page_config here)
    # -------------------------------------------------
    st.title("Longevity Chatbot 🧬")

    # -------------------------------------------------
    # Paths
    # -------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "longevity_responses.json")

    # -------------------------------------------------
    # Load limited-mode data once
    # -------------------------------------------------
    if "stemmed_data" not in st.session_state:
        st.session_state.stemmed_data = load_limited_data(DATA_PATH)

    # -------------------------------------------------
    # Initialize session state (SAFE)
    # -------------------------------------------------
    defaults = {
        "chat_history": [],
        "llm_questions_left": 2,          # 🔧 demo limit
        "use_api": False,
        "awaiting_new_api_key": False,
        "choose_next_action": False,
        "welcome_shown": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # -------------------------------------------------
    # Determine API availability
    # -------------------------------------------------
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()

    if api_key and not st.session_state.welcome_shown:
        st.session_state.use_api = True

        st.info(
            "🤖✨ **LLM Mode Activated — Welcome to the Longevity Chatbot**\n\n"
            "You’re chatting with an AI assistant powered by a large language model.\n\n"
            "**Topics you can explore:**\n"
            "• 🥗 Nutrition & diet\n"
            "• 🏃 Physical activity\n"
            "• 😴 Sleep optimization\n"
            "• 🧠 Brain health\n"
            "• 💆 Stress management\n"
            "• 🤝 Social connection & purpose\n"
            "• 🦠 Gut health & inflammation\n"
            "• 🧬 Healthy aging\n\n"
            "ℹ️ You have a limited number of LLM-powered questions in this session.\n"
            "Once the limit is reached, you can switch to Limited Mode or enter your own GOOGLE / GENAI API key.\n\n"
            "💬 Ask a question to get started."
        )

        st.session_state.welcome_shown = True

    elif not api_key:
        st.session_state.use_api = False

    # -------------------------------------------------
    # Status badge (LLM remaining)
    # -------------------------------------------------
    if st.session_state.use_api and st.session_state.llm_questions_left > 0:
        st.markdown(
            f"""
            <div style="
                background-color:#e6f4ea;
                padding:10px;
                border-radius:8px;
                margin-bottom:10px;
                font-weight:600;
            ">
            🤖 LLM Questions Remaining: {st.session_state.llm_questions_left}
            </div>
            """,
            unsafe_allow_html=True
        )

    # -------------------------------------------------
    # Limited Mode banner
    # -------------------------------------------------
    if not st.session_state.use_api:
        st.info(
            "📘 **Limited Mode Active**\n\n"
            "Responses are powered by curated, keyword-based knowledge.\n\n"
            "**Try asking about:**\n"
            "- 🥗 Diet & nutrition\n"
            "- 🏃 Exercise & movement\n"
            "- 😴 Sleep\n"
            "- 💆 Stress management\n"
            "- 🧠 Brain health\n"
            "- 🤝 Social connection\n"
            "- 🦠 Gut health\n"
            "- 🧬 Genetics\n\n"
            "Tip: Clear keywords work best."
        )

    # -------------------------------------------------
    # Quota decision UI
    # -------------------------------------------------
    if st.session_state.choose_next_action:
        st.markdown("### 🔐 Choose how to continue:")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔑 Enter new GOOGLE / GENAI API key"):
                st.session_state.awaiting_new_api_key = True
                st.session_state.choose_next_action = False
                st.rerun()

        with col2:
            if st.button("📘 Switch to Limited Mode"):
                st.session_state.use_api = False
                st.session_state.llm_questions_left = -1
                st.session_state.choose_next_action = False
                st.session_state.awaiting_new_api_key = False

                st.session_state.chat_history.append(
                    (
                        "Chatbot",
                        "📘 **Limited Mode Activated.** You can continue using keyword-based questions."
                    )
                )
                st.rerun()

    # -------------------------------------------------
    # Chat input
    # -------------------------------------------------
    with st.form("chat_form", clear_on_submit=True):
        if st.session_state.awaiting_new_api_key:
            user_input = st.text_input(
                "Enter your Google / GenAI API key:",
                type="password"
            )
        else:
            user_input = st.text_input(
                "You:",
                placeholder="Ask a longevity question..."
            )

        submit = st.form_submit_button("Send")

    # -------------------------------------------------
    # Handle input
    # -------------------------------------------------
    if submit and user_input:

        # --- API key submission ---
        if st.session_state.awaiting_new_api_key:
            new_key = user_input.strip()

            if new_key:
                os.environ["GOOGLE_API_KEY"] = new_key
                st.session_state.use_api = True
                st.session_state.llm_questions_left = 2
                st.session_state.awaiting_new_api_key = False

                st.session_state.chat_history.append(
                    ("Chatbot", "✅ New API key accepted. LLM Mode reactivated.")
                )
            else:
                st.session_state.use_api = False
                st.session_state.llm_questions_left = -1
                st.session_state.awaiting_new_api_key = False

                st.session_state.chat_history.append(
                    ("Chatbot", "⚠️ No API key entered. Staying in Limited Mode.")
                )

            st.rerun()

        # --- Normal chat flow ---
        else:
            if st.session_state.use_api and st.session_state.llm_questions_left > 0:
                try:
                    bot_message = generate_llm_response(user_input)
                    st.session_state.llm_questions_left -= 1

                    if st.session_state.llm_questions_left == 0:
                        st.session_state.choose_next_action = True
                        bot_message += (
                            "\n\n⚠️ **Demo LLM limit reached.**\n"
                            "This demo showcases controlled LLM usage.\n"
                            "You can continue exploring with Limited Mode or add your own API key."

                        )

                except Exception as e:
                    st.session_state.use_api = False
                    st.session_state.llm_questions_left = -1
                    bot_message = (
                        f"⚠️ API error: {e}\n\n"
                        "Switched to Limited Mode."
                    )
            else:
                bot_message = get_limited_response(
                    user_input,
                    st.session_state.stemmed_data
                )

            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Chatbot", bot_message))
            st.rerun()

    # -------------------------------------------------
    # Render conversation
    # -------------------------------------------------
    st.markdown("### 💬 Conversation")

    for speaker, message in reversed(st.session_state.chat_history):
        if speaker == "You":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**Chatbot:** {message}")

    st.markdown("---")


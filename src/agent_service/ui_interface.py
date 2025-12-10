from typing import Dict, List

import gradio as gr
import requests

# Agent service URL
AGENT_API_URL = "http://localhost:8003"

# Global state
chat_history: List[Dict[str, str]] = []
status_message = "🟢 Ready"


def analyze_message(
    message: str, history: List[Dict[str, str]]
) -> tuple[List[Dict[str, str]], str, str]:
    """Analyze message with enhanced UX feedback"""
    global status_message

    if not message or not message.strip():
        return history, "", status_message

    status_message = "🔍 Analyzing sentiment..."

    try:
        # Show user message and typing indicator
        new_history = history.copy()
        new_history.append({"role": "user", "content": message})
        new_history.append({"role": "assistant", "content": "⏳ Processing..."})

        # Call the Agent API
        response = requests.post(
            f"{AGENT_API_URL}/predict", json={"text": message}, timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            predicted_class = result.get("predicted_class", "Unknown")
            confidence = result.get("confidence", 0)

            # Enhanced response with confidence and emoji
            emoji_map = {
                "positive": "😊",
                "negative": "😞",
                "neutral": "😐",
                "urgent": "🚨",
                "complaint": "⚠️",
                "question": "❓",
                "support": "🆘",
                "info": "ℹ️",
            }
            emoji = emoji_map.get(predicted_class.lower(), "🤖")

            final_response = (
                f"{emoji} **{predicted_class.title()}** (Confidence: {confidence:.1%})"
            )

            # Replace typing indicator
            new_history[-1] = {"role": "assistant", "content": final_response}
            status_message = "✅ Ready"

        else:
            error_msg = "❌ Service temporarily unavailable. Please try again."
            new_history[-1] = {"role": "assistant", "content": error_msg}
            status_message = "⚠️ Service error"

    except requests.exceptions.ConnectionError:
        error_msg = "🔌 Backend service offline. Ensure it's running on port 8003."
        new_history[-1] = {"role": "assistant", "content": error_msg}
        status_message = "🔌 Disconnected"
    except requests.exceptions.Timeout:
        error_msg = "⏱️ Request timeout. Please try again."
        new_history[-1] = {"role": "assistant", "content": error_msg}
        status_message = "⏳ Slow response"
    except Exception as e:
        error_msg = f"⚠️ Unexpected error: {str(e)[:100]}"
        new_history[-1] = {"role": "assistant", "content": error_msg}
        status_message = "⚠️ Error"

    return new_history, "", status_message


def clear_chat() -> tuple[list, str, str]:
    """Clear conversation and reset state"""
    global chat_history, status_message
    chat_history = []
    status_message = "🟢 Ready"
    return [], "", status_message


css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Global background with orange–purple gradient */
body {
    background: radial-gradient(circle at top left, #ff9f68 0%, #ff6b35 15%, #4b164c 55%, #1a1024 100%) !important;
}

.gradio-container {
    width: 90vw !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 2rem 1rem !important;
    background: transparent !important;
}

/* Header with orange–purple text gradient */
.header-section {
    background: rgba(25, 10, 40, 0.9) !important;
    backdrop-filter: blur(20px);
    border-radius: 18px;
    border: 1px solid rgba(255, 140, 66, 0.45);
    padding: 1.75rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 18px 35px rgba(0,0,0,0.35);
    text-align: center;
}

/* Status bar with soft gradient */
.status-bar {
    background: linear-gradient(135deg, rgba(255, 140, 66, 0.18), rgba(168, 85, 247, 0.18)) !important;
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255, 140, 66, 0.5);
    border-radius: 10px;
    padding: 0.75rem 1.25rem !important;
    margin-bottom: 1.25rem;
    font-weight: 500;
    color: #fde7ff !important;
    font-size: 0.9rem;
}

/* Chatbot container: compact, glassy, with border gradient */
#chatbot {
    background: rgba(20, 9, 35, 0.92) !important;
    backdrop-filter: blur(20px) !important;
    border-radius: 18px !important;
    box-shadow: 0 18px 40px rgba(0,0,0,0.55) !important;
    height: 380px !important;
    border: 1px solid transparent !important;
    background-image:
        linear-gradient(135deg, #1a0f25, #1a0f25),
        linear-gradient(135deg, #ff6b35, #ff8c42, #a855f7, #7c3aed);
    background-origin: border-box;
    background-clip: padding-box, border-box;
}

/* Messages */
.message {
    padding: 1.1rem 1.3rem !important;
    margin: 0.6rem 0 !important;
    border-radius: 14px !important;
    line-height: 1.5;
    font-size: 0.92rem;
}

.user {
    background: linear-gradient(135deg, #ff6b35, #ff8c42) !important;
    color: #fff8f2 !important;
    margin-left: 20% !important;
    box-shadow: 0 10px 28px rgba(255, 107, 53, 0.35);
}

.assistant {
    background: radial-gradient(circle at top left, rgba(168,85,247,0.85), rgba(76,29,149,0.9)) !important;
    color: #fdf2ff !important;
    margin-right: 20% !important;
    border-left: 3px solid #ffb374;
    box-shadow: 0 10px 30px rgba(88, 28, 135, 0.6);
}

/* Input area */
.form {
    background: rgba(18, 9, 30, 0.9) !important;
    backdrop-filter: blur(20px);
    border-radius: 20px !important;
    border: 1px solid rgba(255, 140, 66, 0.35);
    padding: 0.9rem 1rem !important;
    margin-top: 0.75rem;
}

textarea {
    background: transparent !important;
    border: none !important;
    color: #fde7ff !important;
    font-size: 0.95rem !important;
}

textarea::placeholder {
    color: rgba(252, 211, 255, 0.6) !important;
}

/* Buttons: orange–purple gradient */
button {
    border-radius: 999px !important;
    padding: 0.7rem 1.4rem !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    border: none !important;
    cursor: pointer;
    transition: all 0.25s ease;
}

/* Send button */
button[type="submit"] {
    background-image: linear-gradient(135deg, #ff6b35, #ff8c42, #a855f7) !important;
    color: white !important;
    box-shadow: 0 10px 26px rgba(255, 107, 53, 0.45);
}

button[type="submit"]:hover {
    transform: translateY(-2px) scale(1.03);
    box-shadow: 0 16px 36px rgba(168, 85, 247, 0.5);
}

/* Clear button */
.clear-btn {
    background: rgba(64, 25, 104, 0.85) !important;
    color: #fde7ff !important;
    border: 1px solid rgba(255, 140, 66, 0.45) !important;
}

.clear-btn:hover {
    background: radial-gradient(circle at top left, #a855f7, #ff6b35) !important;
    color: #fffaf2 !important;
}

/* Examples panel */
.examples-panel {
    background: rgba(18, 9, 30, 0.9) !important;
    backdrop-filter: blur(16px);
    border-radius: 14px;
    border: 1px solid rgba(168, 85, 247, 0.45);
    padding: 1.1rem;
    margin-top: 1.5rem;
}

#examples_container button {
    background: linear-gradient(135deg, rgba(255, 140, 66, 0.9), rgba(168, 85, 247, 0.95)) !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.2rem !important;
    margin: 0.25rem !important;
    border: none !important;
    color: #fffaf5 !important;
    font-size: 0.85rem !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #ff6b35, #a855f7);
    border-radius: 10px;
}
::-webkit-scrollbar-track {
    background: rgba(20, 8, 30, 0.8);
}
"""

with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
    <div class="header-section">
        <div style="text-align: center;">
            <div style="
                display: inline-block;
                background: linear-gradient(135deg, #ff6b35, #ff8c42, #a855f7, #7c3aed);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 1.9rem;
                font-weight: 800;
                letter-spacing: -0.02em;
                margin-bottom: 0.4rem;
            ">
                🤖 Callcenter AI </div>
            <p style="
                color: rgba(255, 240, 255, 0.85);
                font-size: 0.98rem;
                margin: 0 auto;
                max-width: 430px;
            ">
                Real-time customer IT classification  </p>
        </div>
    </div>
    """
    )

    status = gr.HTML(value=status_message, elem_classes=["status-bar"])

    chatbot = gr.Chatbot(
        label="💬 Conversation", height=380, show_label=False, elem_id="chatbot"
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="💬 Enter customer message...",
            show_label=False,
            lines=1,
            container=False,
            scale=4,
        )
        submit = gr.Button("➤ Send", scale=1)

    clear = gr.Button("🔄 New Chat", elem_classes=["clear-btn"])

    submit.click(analyze_message, [msg, chatbot], [chatbot, msg, status])
    msg.submit(analyze_message, [msg, chatbot], [chatbot, msg, status])
    clear.click(clear_chat, outputs=[chatbot, msg, status])

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False,
    )

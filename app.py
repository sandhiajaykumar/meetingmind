import os
import json
import tempfile

import streamlit as st

from dotenv import load_dotenv

from google import genai

from src.rag import (
    create_vectorstore
)

from src.agent import (
    ask_meeting_agent
)


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

api_key = os.getenv(
    "GEMINI_API_KEY"
)

if not api_key:

    st.error(
        "GEMINI_API_KEY is missing from your .env file."
    )

    st.stop()


client = genai.Client(
    api_key=api_key
)


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="MeetingMind",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🧠 MeetingMind")

st.markdown(
    """
    ### AI Meeting Intelligence & Agent

    Transform meeting audio into actionable intelligence.

    **Audio → Gemini → Analysis → FAISS RAG → LangGraph Agent**
    """
)


# ============================================================
# SESSION STATE
# ============================================================

if "analysis" not in st.session_state:

    st.session_state.analysis = None


if "transcript" not in st.session_state:

    st.session_state.transcript = None


# ============================================================
# AUDIO UPLOAD
# ============================================================

st.header("🎙️ Upload Meeting")

audio = st.file_uploader(
    "Upload your meeting recording",
    type=[
        "mp3",
        "wav",
        "m4a",
        "mp4"
    ]
)


# ============================================================
# ANALYZE MEETING
# ============================================================

if st.button(
    "🚀 Analyze Meeting",
    type="primary"
):

    if audio is None:

        st.warning(
            "Please upload a meeting audio file."
        )

        st.stop()


    # --------------------------------------------------------
    # Save temporary audio file
    # --------------------------------------------------------

    file_extension = os.path.splitext(
        audio.name
    )[1]

    if not file_extension:

        file_extension = ".wav"


    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=file_extension
    ) as temp_file:

        temp_file.write(
            audio.read()
        )

        audio_path = temp_file.name


    try:

        # ====================================================
        # UPLOAD AUDIO TO GEMINI
        # ====================================================

        with st.spinner(
            "📤 Uploading meeting audio..."
        ):

            uploaded_file = client.files.upload(
                file=audio_path
            )


        # ====================================================
        # ANALYSIS PROMPT
        # ====================================================

        prompt = """
You are an expert meeting intelligence assistant.

Analyze the attached meeting audio carefully.

Your first priority is accurate transcription.

Preserve exactly as much as possible:

- Speaker names
- Technical terms
- Product names
- Framework names
- Database names
- Numbers
- Dates
- Deadlines

Do not replace technical terms with generic words.

For example:
FAISS must remain FAISS.
LangChain must remain LangChain.
LangGraph must remain LangGraph.

Then extract:

1. Complete transcript
2. Concise meeting summary
3. Important decisions
4. Action items
5. Risks or unresolved issues

For every action item identify:

- task
- owner
- deadline

Do NOT invent information.

If an owner is unknown:
"Unknown"

If a deadline is unknown:
"Not specified"

Return ONLY valid JSON.

Do not use Markdown.
Do not use ```json.
Do not add explanations outside the JSON.

Use exactly this structure:

{
    "transcript": "string",
    "summary": "string",
    "decisions": [],
    "action_items": [
        {
            "task": "string",
            "owner": "string",
            "deadline": "string"
        }
    ],
    "risks": []
}
"""


        # ====================================================
        # GEMINI AUDIO ANALYSIS
        # ====================================================

        with st.spinner(
            "🧠 Gemini is analyzing the meeting..."
        ):

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    uploaded_file,
                    prompt
                ]
            )


        # ====================================================
        # CLEAN GEMINI RESPONSE
        # ====================================================

        raw_response = response.text.strip()


        # Remove Markdown code fences

        if raw_response.startswith(
            "```"
        ):

            raw_response = raw_response.replace(
                "```json",
                ""
            )

            raw_response = raw_response.replace(
                "```",
                ""
            )

            raw_response = raw_response.strip()


        # ====================================================
        # PARSE JSON
        # ====================================================

        result = json.loads(
            raw_response
        )


        # ====================================================
        # STORE SESSION DATA
        # ====================================================

        st.session_state.analysis = result

        st.session_state.transcript = (
            result["transcript"]
        )


        # ====================================================
        # SAVE CURRENT MEETING
        # ====================================================

        os.makedirs(
            "data",
            exist_ok=True
        )


        with open(
            "data/current_meeting.json",
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                result,
                file,
                indent=4,
                ensure_ascii=False
            )


        # ====================================================
        # ADD MEETING TO FAISS
        # ====================================================

        with st.spinner(
            "🧠 Adding meeting to vector memory..."
        ):

            create_vectorstore(
                transcript=result["transcript"],
                meeting_id=audio.name
            )


        st.success(
            "✅ Meeting analyzed and stored in memory!"
        )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except json.JSONDecodeError:

        st.error(
            "Gemini returned invalid JSON."
        )

        st.code(
            response.text,
            language="text"
        )


    except Exception as error:

        st.error(
            "An error occurred."
        )

        st.exception(error)


    finally:

        # ----------------------------------------------------
        # Delete temporary audio
        # ----------------------------------------------------

        if os.path.exists(
            audio_path
        ):

            os.remove(
                audio_path
            )


# ============================================================
# DISPLAY MEETING ANALYSIS
# ============================================================

if st.session_state.analysis:

    result = st.session_state.analysis


    st.divider()


    # ========================================================
    # TRANSCRIPT
    # ========================================================

    st.header("📝 Transcript")

    st.text_area(
        "Meeting Transcript",
        result.get(
            "transcript",
            ""
        ),
        height=250
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    st.header("📋 Summary")

    st.info(
        result.get(
            "summary",
            "No summary available."
        )
    )


    # ========================================================
    # DECISIONS
    # ========================================================

    st.header("✅ Decisions")

    decisions = result.get(
        "decisions",
        []
    )


    if decisions:

        for decision in decisions:

            st.write(
                f"• {decision}"
            )

    else:

        st.info(
            "No major decisions detected."
        )


    # ========================================================
    # ACTION ITEMS
    # ========================================================

    st.header("📌 Action Items")

    action_items = result.get(
        "action_items",
        []
    )


    if action_items:

        for index, item in enumerate(
            action_items,
            start=1
        ):

            st.markdown(
                f"### {index}. {item.get('task', 'Unknown task')}"
            )


            col1, col2 = st.columns(2)


            with col1:

                st.write(
                    "👤 **Owner:** "
                    + item.get(
                        "owner",
                        "Unknown"
                    )
                )


            with col2:

                st.write(
                    "📅 **Deadline:** "
                    + item.get(
                        "deadline",
                        "Not specified"
                    )
                )


            st.divider()


    else:

        st.info(
            "No action items detected."
        )


    # ========================================================
    # RISKS
    # ========================================================

    st.header(
        "⚠️ Risks / Unresolved Issues"
    )


    risks = result.get(
        "risks",
        []
    )


    if risks:

        for risk in risks:

            st.warning(
                risk
            )

    else:

        st.success(
            "No major risks detected."
        )


# ============================================================
# AI AGENT
# ============================================================

st.divider()

st.header(
    "🤖 Ask MeetingMind Agent"
)

st.write(
    """
    Ask questions about your meeting.

    The LangGraph agent decides which tool
    should be used to answer your question.
    """
)


question = st.text_input(
    "Your question",
    placeholder=(
        "Example: What are the pending tasks?"
    )
)


if st.button(
    "🤖 Ask Agent"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()


    with st.spinner(
        "🤖 MeetingMind Agent is thinking..."
    ):

        try:

            answer = ask_meeting_agent(
                question
            )


            st.subheader(
                "💡 Answer"
            )


            st.success(
                answer
            )


        except Exception as error:

            st.error(
                "Agent error occurred."
            )

            st.exception(error)


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.caption(
    "MeetingMind | Gemini + LangChain + FAISS + RAG + LangGraph"
)
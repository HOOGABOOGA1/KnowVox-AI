import os
import time
import re
import requests
from typing import List, Dict, Any, Optional
from app.core.config import settings


class RAGService:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        if self.api_key == "your_gemini_api_key_here":
            self.api_key = ""
        self.model = model or settings.DEFAULT_MODEL

    def _get_api_key(self) -> str:
        key = self.api_key or settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        if not key or key == "your_gemini_api_key_here":
            from app.core.config import base_dir
            for p in [base_dir / ".env", base_dir.parent / ".env", base_dir / "voicerag" / ".env"]:
                if p.exists():
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            for line in f:
                                if line.startswith("GEMINI_API_KEY="):
                                    k = line.split("=", 1)[1].strip()
                                    if k and k != "your_gemini_api_key_here":
                                        settings.GEMINI_API_KEY = k
                                        self.api_key = k
                                        return k
                    except Exception:
                        pass
        return key

    def _get_model(self) -> str:
        return self.model or settings.DEFAULT_MODEL

    def generate_answer(
        self,
        query: str,
        context_chunks: Optional[List[Dict[str, Any]]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generates a refined, context-grounded response with Career OS profile awareness.
        """
        current_api_key = self._get_api_key()
        sources_set = set()

        # 1. Format User Profile Context
        profile_str = "No user profile set."
        user_name = "User"
        if user_profile:
            user_name = user_profile.get("name") or "User"
            edu = user_profile.get("education") or "Not specified"
            skills = user_profile.get("skills") or "Not specified"
            goal = user_profile.get("career_goal") or "Not specified"
            exp = user_profile.get("experience") or "Not specified"
            loc = user_profile.get("location") or "Not specified"
            interests = user_profile.get("interests") or "Not specified"
            profile_str = f"""
- User Name: {user_name}
- Education: {edu}
- Skills: {skills}
- Career Goal: {goal}
- Experience / Projects: {exp}
- Interests: {interests}
- Location: {loc}
"""

        # 2. Format Document Context Chunks
        context_parts = []
        if context_chunks:
            for c in context_chunks:
                meta = c.get("metadata", {})
                source = meta.get("source", "document")
                page = meta.get("page", 1)
                sources_set.add(f"{source} (Page {page})")
                context_parts.append(f"[Document: {source} | Page {page}]\n{c['text']}")

        context_str = "\n\n".join(context_parts) if context_parts else "No specific documents attached to this query."

        # 3. Format Conversation History
        history_str = ""
        if conversation_history:
            formatted_msgs = []
            for msg in conversation_history[-4:]:
                role = msg.get("role", "user").capitalize()
                content = msg.get("content", "").strip()
                if len(content) > 350:
                    content = content[:350] + "..."
                formatted_msgs.append(f"{role}: {content}")
            history_str = "\n".join(formatted_msgs)

        # 4. Check for Detailed / Elaboration Request
        elaboration_keywords = [
            "explain", "detail", "details", "elaborate", "expand", "brief",
            "more", "depth", "deep", "roadmap", "guide", "step", "steps",
            "breakdown", "list", "thorough", "thoroughly", "full", "complete",
            "long", "big", "describe", "summary", "summarize", "everything",
            "teach", "overview", "how to", "why", "strategy", "plan", "interview",
            "tell me more", "break it down", "all", "in detail", "give a brief",
            "explain in detail", "give more info", "notes", "cheatsheet", "study guide",
            "tutorial", "revision", "concepts"
        ]
        q_lower = query.lower()
        is_detailed_requested = any(kw in q_lower for kw in elaboration_keywords)

        if is_detailed_requested:
            length_instruction = """- DETAILED & COMPREHENSIVE MODE ACTIVE:
  The user is asking for notes, an in-depth explanation, roadmap, or comprehensive breakdown.
  Provide a rich, well-structured, multi-section response with clear section headings, actionable code/bullet points, and complete details."""
            max_tokens = 2500
        else:
            length_instruction = """- CONCISE SPOKEN MODE (DEFAULT FOR ULTRA-FAST VOICE RESPONSES):
  Keep your answer crisp, direct, and conversational (typically 2 to 3 punchy sentences, under 50 words).
  Deliver the essential factual answer immediately. If the user wants a full breakdown, they will ask to "explain", "notes", "detail", or "expand"."""
            max_tokens = 320

        # 5. Refined Prompt with Career OS Profile & Grounded RAG
        prompt = f"""You are KnowVox, an intelligent, empathetic, articulate AI Career Copilot and Voice Knowledge Assistant.

USER'S CAREER OS PROFILE:
{profile_str}

ATTACHED DOCUMENTS CONTEXT:
{context_str}

CONVERSATION HISTORY:
{history_str if history_str else "None"}

USER QUESTION / PROMPT:
{query}

BEHAVIOR GUIDELINES:
1. {length_instruction}
2. INTENT ROUTING:
   - If the user asks about themselves, their career, roadmap, skills, education, or "my career info", "who am I", "what work should I do", answer DIRECTLY and PERSONALLY using their Career OS Profile facts. Do NOT force unrelated document context (such as PDFs about space science or other topics) into personal career questions.
   - If the user asks for technical notes, tutorials, or study materials (e.g. "give me python notes", "dsa notes"), provide comprehensive, high-quality technical notes with key concepts and syntax examples.
   - If the user asks a factual question about their uploaded documents, notes, or files, ground your answer directly in the facts from ATTACHED DOCUMENTS CONTEXT with citations.
3. Maintain an articulate, encouraging, and natural spoken tone suitable for real-time voice output.
4. Do NOT append raw citation tags like "Source: doc.pdf" inside the speech text as sources are rendered separately in the UI.

ANSWER:"""

        # 6. Generate response with cascading Gemini models & retry
        if current_api_key:
            target_model = self._get_model()
            candidate_models = [
                "gemini-3.1-flash-lite",
                target_model,
                "gemini-flash-lite-latest",
                "gemini-3.6-flash"
            ]
            unique_models = []
            for m in candidate_models:
                if m and m not in unique_models and m not in ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-1.5-flash", "gemini-2.0-flash"]:
                    unique_models.append(m)

            for model_name in unique_models:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={current_api_key}"
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.3 if not is_detailed_requested else 0.4,
                        "topP": 0.95,
                        "maxOutputTokens": max_tokens
                    }
                }
                try:
                    response = requests.post(url, json=payload, timeout=12.0)
                    if response.status_code == 200:
                        data = response.json()
                        candidates = data.get("candidates", [])
                        if candidates and "content" in candidates[0] and "parts" in candidates[0]["content"]:
                            parts = candidates[0]["content"]["parts"]
                            answer_text = "".join([p.get("text", "") for p in parts]).strip()
                            if answer_text:
                                return {
                                    "answer": answer_text,
                                    "sources": sorted(list(sources_set))
                                }
                    elif response.status_code == 429:
                        time.sleep(0.2)
                        continue
                except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout):
                    # Immediately break to offline mode if connection is down
                    break
                except Exception:
                    continue

        # 7. Smart Local / Offline Fallback
        return self._generate_smart_fallback_answer(query, context_chunks, user_profile, sources_set)

    def _generate_smart_fallback_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        user_profile: Optional[Dict[str, Any]],
        sources_set: set
    ) -> Dict[str, Any]:
        """Synthesize answer when offline or when the cloud API key/endpoint is unavailable."""
        q_lower = query.lower().strip()
        user_name = (user_profile.get("name") if user_profile else None) or "there"
        cap_name = user_name.capitalize()
        edu = (user_profile.get("education") if user_profile else None) or "Professional Degree"
        skills = (user_profile.get("skills") if user_profile else None) or "Core Domain Skills"
        goal = (user_profile.get("career_goal") if user_profile else None) or "Professional Career"
        exp = (user_profile.get("experience") if user_profile else None) or "Projects & Experience"
        g_lower = goal.lower()

        has_negative_interview = any(neg in q_lower for neg in ["dont want", "don't want", "no interview", "not interview", "stop interview", "cancel interview", "not an interview", "instead", "dont do"])

        # 1. Universal Domain-Aware Mock Interview Mode
        if any(w in q_lower for w in ["mock interview", "practice interview", "start interview", "run a mock interview", "interview question", "ask me a question"]) and not has_negative_interview:
            if any(k in g_lower for k in ["doctor", "neuro", "medic", "surgeon", "clinic", "health", "mbbs"]):
                answer = (
                    f"🩺 **Clinical Case Study Interview for {goal}**\n\n"
                    f"**Scenario for {cap_name}:**\n"
                    f"\"A 62-year-old patient presents with acute onset right-sided weakness and aphasia starting 90 minutes ago. Blood pressure is 185/105 mmHg.\"\n\n"
                    f"🎙️ **Questions:**\n"
                    f"1. What is your immediate priority and non-contrast CT protocol?\n"
                    f"2. What are the inclusion/exclusion criteria for IV thrombolysis (rtPA / Tenecteplase)?\n\n"
                    f"Take 30 seconds to formulate your clinical diagnosis, then speak or type your answer!"
                )
            elif any(k in g_lower for k in ["law", "attorney", "legal", "advocate", "m&a"]):
                answer = (
                    f"⚖️ **Corporate Law & Deal Structuring Interview for {goal}**\n\n"
                    f"**Scenario for {cap_name}:**\n"
                    f"\"In a cross-border Share Purchase Agreement (SPA), the target company fails to disclose a pending material tax litigation before closing.\"\n\n"
                    f"🎙️ **Questions:**\n"
                    f"1. How would you structure the Indemnity and Reps & Warranties clauses to protect the buyer?\n"
                    f"2. What is the difference between an Indemnity Escrow and a Specific Indemnity Basket?\n\n"
                    f"Formulate your legal analysis and respond whenever ready!"
                )
            elif any(k in g_lower for k in ["finance", "bank", "invest", "trading", "analyst"]):
                answer = (
                    f"📈 **Investment Banking & Valuation Interview for {goal}**\n\n"
                    f"**Question for {cap_name}:**\n"
                    f"\"Walk me through how a $10 increase in depreciation affects all three financial statements, and how does it impact Enterprise Value in a DCF model?\"\n\n"
                    f"🎙️ **Key Points to Cover:**\n"
                    f"• Net Income adjustment with tax shield\n"
                    f"• Cash Flow from Operations change\n"
                    f"• Impact on Unlevered Free Cash Flow and WACC\n\n"
                    f"Take 30 seconds to structure your breakdown, then answer!"
                )
            elif any(k in g_lower for k in ["design", "ui", "ux", "product design"]):
                answer = (
                    f"🎨 **UI/UX Product Design Interview for {goal}**\n\n"
                    f"**Scenario for {cap_name}:**\n"
                    f"\"You are tasked with redesigning a high-dropoff checkout flow for a mobile application.\"\n\n"
                    f"🎙️ **Questions:**\n"
                    f"1. How would you conduct user research and identify friction points without relying purely on assumptions?\n"
                    f"2. How do you balance business constraints with user accessibility (WCAG 2.1 standards)?\n\n"
                    f"Explain your design thinking framework when ready!"
                )
            else:
                answer = (
                    f"🎯 **Technical Interview for {goal}**\n\n"
                    f"**Question for {cap_name}:**\n"
                    f"\"Based on your background in **{skills}**, can you explain the architectural trade-offs you make when designing a system for high scalability versus low latency?\"\n\n"
                    f"🎙️ **Key Areas to Address:**\n"
                    f"• Core data structures and algorithmic complexity\n"
                    f"• Concurrency models and caching strategies\n"
                    f"• Reliability and fault tolerance\n\n"
                    f"Take 30 seconds to formulate your answer, then speak or type!"
                )
            return {"answer": answer, "sources": []}

        # 2. Career OS Profile Overview & Notes
        is_profile_notes = any(w in q_lower for w in [
            "notes about my career", "notes on my career", "career profile notes", "profile notes",
            "notes about my profile", "notes on my profile", "my career profile", "about my career profile",
            "what is in my profile", "who am i", "my details", "my background", "my skills", "my education",
            "tell me about my career", "tell me about my profile", "summarize my profile", "my career info",
            "career profile"
        ])
        if is_profile_notes or (any(w in q_lower for w in ["profile", "career"]) and any(w in q_lower for w in ["note", "notes", "summary", "details", "info", "overview", "give me"])):
            loc = (user_profile.get("location") if user_profile else None) or "Global"
            interests = (user_profile.get("interests") if user_profile else None) or "Professional Excellence"
            answer = (
                f"👤 **Career OS Profile Notes**\n\n"
                f"• **Name:** {cap_name}\n"
                f"• **Target Career Goal:** {goal}\n"
                f"• **Education:** {edu}\n"
                f"• **Primary Skills:** {skills}\n"
                f"• **Experience:** {exp}\n"
                f"• **Interests & Domain:** {interests}\n"
                f"• **Location:** {loc}\n\n"
                f"📌 **Key Strategic Takeaways:**\n"
                f"1. **Target Alignment:** Pursuing role as **{goal}** backed by your foundation in **{edu}**.\n"
                f"2. **Core Specialization:** Strong hands-on focus in **{skills}**.\n"
                f"3. **Actionable Roadmap:** Build real-world project portfolios, document methodologies, and practice scenario-based technical interviews to accelerate your transition to {goal}!\n\n"
                f"💡 *Ask for 'career roadmap' or 'start mock interview' to practice anytime.*"
            )
            return {"answer": answer, "sources": []}

        # 3. Universal Domain-Aware Career Roadmap
        profile_keywords = [
            "my goal", "next step", "next steps", "roadmap", "strategy",
            "advice", "guidance", "recommendation", "suggest", "what work", "work should",
            "what should i", "do next", "doing next", "steps", "my step", "path",
            "direction", "start", "prepare", "job", "career goal", "my career",
            "career roadmap", "generate ai career roadmap"
        ]
        if any(w in q_lower for w in profile_keywords):
            if any(k in g_lower for k in ["doctor", "neuro", "medic", "surgeon", "clinic", "health", "mbbs"]):
                answer = (
                    f"🩺 **Clinical Excellence Roadmap for {cap_name}**\n"
                    f"**Goal:** {goal} | **Specialization:** {skills}\n\n"
                    f"### Phase 1: Clinical Mastery & Diagnostic Protocol (Months 1–2)\n"
                    f"• Master advanced neuro-imaging (MRI Diffusion, CT Angiography) and emergency stroke protocols.\n"
                    f"• Complete clinical case audits and refine differential diagnosis accuracy.\n\n"
                    f"### Phase 2: Board Exam & Research Publication (Months 3–4)\n"
                    f"• High-yield revision on clinical pharmacology, neuroplasticity, and clinical trials.\n"
                    f"• Draft and submit an observational clinical paper or case report to a peer-reviewed journal.\n\n"
                    f"### Phase 3: Residency & Fellowship Placement (Months 5–6)\n"
                    f"• Prepare for structured clinical examinations (OSCE) and consultant interviews.\n\n"
                    f"Would you like to run a clinical mock diagnosis case now?"
                )
            elif any(k in g_lower for k in ["law", "attorney", "legal", "advocate", "m&a"]):
                answer = (
                    f"⚖️ **Corporate Legal Counsel Roadmap for {cap_name}**\n"
                    f"**Goal:** {goal} | **Domain:** {skills}\n\n"
                    f"### Phase 1: Due Diligence & Deal Structuring (Months 1–2)\n"
                    f"• Master Virtual Data Room (VDR) legal audits, title verification, and regulatory compliance.\n"
                    f"• Draft boilerplate and bespoke clauses for Share Purchase Agreements (SPAs) and JV terms.\n\n"
                    f"### Phase 2: Antitrust, FDI & Cross-Border Nuance (Months 3–4)\n"
                    f"• Study cross-border merger control thresholds, FDI regulations, and tax treaties (DTAA).\n"
                    f"• Build a Precedent Clause Library for international arbitration seats (Singapore, London).\n\n"
                    f"### Phase 3: Senior Associate Deal Execution (Months 5–6)\n"
                    f"• Lead negotiation simulations on Indemnity Caps, Baskets, and Conditions Precedent (CPs).\n\n"
                    f"Would you like to analyze an M&A contract clause or practice an interview?"
                )
            elif any(k in g_lower for k in ["finance", "bank", "invest", "trading", "analyst"]):
                answer = (
                    f"📈 **Investment Banking & Finance Roadmap for {cap_name}**\n"
                    f"**Goal:** {goal} | **Domain:** {skills}\n\n"
                    f"### Phase 1: Financial Modeling & Valuation (Months 1–2)\n"
                    f"• Build dynamic 3-Statement financial models, Discounted Cash Flow (DCF), and Comps in Excel.\n"
                    f"• Master LBO debt waterfall modeling and sensitivity tables.\n\n"
                    f"### Phase 2: Pitchbook & M&A Deal Analysis (Months 3–4)\n"
                    f"• Create real-world investment memos and industry teardowns.\n"
                    f"• Prepare for technical finance interviews (WACC, Enterprise Value, Accretion/Dilution).\n\n"
                    f"### Phase 3: Networking & Superday Recruitment (Months 5–6)\n"
                    f"• Connect with Analysts and Associates at target investment banks.\n\n"
                    f"Would you like to run an Investment Banking valuation mock question?"
                )
            elif any(k in g_lower for k in ["gov", "civil", "upsc", "ssc", "ias", "ips", "public", "policy", "administrat"]):
                answer = (
                    f"🏛️ **Civil Services & Public Administration Roadmap for {cap_name}**\n"
                    f"**Goal:** {goal} | **Subjects:** {skills}\n\n"
                    f"### Phase 1: Foundational Conceptualization (Months 1–3)\n"
                    f"• Master NCERT standard textbooks (Class 6–12) for Indian History, Polity, and Geography.\n"
                    f"• Study M. Laxmikanth's *Indian Polity* and build article-by-article constitutional notes.\n"
                    f"• Begin daily editorial analysis from *The Hindu* or *The Indian Express*.\n\n"
                    f"### Phase 2: Analytical Answer Writing & Optional Mastery (Months 4–6)\n"
                    f"• Select and deeply prepare your Optional Subject ({skills}).\n"
                    f"• Practice structured daily GS Mains answer writing with introduction, multi-dimensional body, and policy-focused conclusion.\n"
                    f"• Build living case studies on NITI Aayog policy reports and Law Commission recommendations.\n\n"
                    f"### Phase 3: Prelims Testing & Interview Simulation (Months 7–9)\n"
                    f"• Solve 50+ full-length sectional and GS Prelims mock papers with strict negative-marking analysis.\n"
                    f"• Participate in personality test panel simulations focusing on administrative neutrality and problem solving.\n\n"
                    f"Would you like a daily study timetable template or an essay topic to practice?"
                )
            else:
                answer = (
                    f"🎯 **Strategic Career Roadmap for {cap_name}**\n"
                    f"**Target Goal:** {goal} | **Domain:** {skills}\n\n"
                    f"### Phase 1: Core Foundation & Professional Milestones (Month 1)\n"
                    f"• Build core competency benchmarks and structure your study/action roadmap in **{skills}**.\n"
                    f"• Create a curated repository of foundational resources, textbooks, and documentation.\n\n"
                    f"### Phase 2: Advanced Competency & Applied Problem Solving (Months 2–3)\n"
                    f"• Deepen practical case studies and master industry-standard workflows.\n"
                    f"• Practice domain-specific assessments and problem-solving simulations.\n\n"
                    f"### Phase 3: Market Entry & Role Placement (Months 4–6)\n"
                    f"• Execute targeted outreach with mentors, organizations, and professional peers.\n\n"
                    f"Would you like to start a targeted mock interview question now?"
                )
            return {"answer": answer, "sources": []}

        # 3. Programming & Tech Stack Guidance
        if any(w in q_lower for w in ["language", "languages", "what language", "which language", "what to learn", "should i learn", "what should i study", "programming language"]):
            answer = (
                f"💻 **Top Recommended Skills & Tools for {goal}**\n\n"
                f"1. **Core Domain Tools:** Prioritize the standard toolset for {goal} (aligned with **{skills}**).\n"
                f"2. **Data & Analytics (Python / SQL):** Essential for automation, metrics analysis, and systems integration.\n"
                f"3. **Collaboration & Systems:** Master industry documentation, version control, and workflow optimization.\n\n"
                f"**Recommendation:** Combining your background in **{skills}** with modern data tools gives you an elite competitive edge!\n\n"
                f"Would you like tailored project recommendations?"
            )
            return {"answer": answer, "sources": []}

        # 4. On-Device Subject Knowledge & Study Notes Engine
        subject_notes = self._get_offline_subject_notes(q_lower, goal, skills, cap_name)
        if subject_notes:
            return {"answer": subject_notes, "sources": []}

        # 5. Document Search / Knowledge Chunks
        if context_chunks:
            return self._generate_offline_answer(query, context_chunks, sources_set)

        # 6. Pure Greetings
        greeting_words = ["hi", "hello", "hey", "good morning", "good evening", "namaste", "greetings", "start"]
        if q_lower in greeting_words or any(q_lower == w for w in greeting_words):
            return {
                "answer": f"Hello {cap_name}! I am KnowVox, your AI Voice & Career Assistant. How can I assist you with your career roadmap for **{goal}**, technical study, or interview practice today?",
                "sources": []
            }

        # 7. General Conversational fallback
        return {
            "answer": (
                f"Hello {cap_name}! Based on your Career OS profile with a goal of **{goal}** and skills in **{skills}**, "
                f"I'm ready to help you optimize your career roadmap, practice mock interviews, or answer questions from your uploaded documents!"
            ),
            "sources": []
        }

    def _generate_offline_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        sources_set: set
    ) -> Dict[str, Any]:
        """Synthesizes rich, detailed, and clear answers offline from retrieved vector store chunks."""
        q_lower = query.lower().strip()
        is_deep_dive = any(w in q_lower for w in [
            "explain more", "more detail", "details", "in detail", "elaborate",
            "tell me more", "expand", "break down", "comprehensive", "deep dive",
            "summarize", "overview", "all laws", "rules", "why", "how"
        ])

        # Extract meaningful search keywords from query
        stop_words = {
            "what", "is", "the", "are", "how", "why", "where", "who", "which",
            "in", "on", "a", "an", "to", "for", "of", "and", "tell", "me", "about",
            "from", "this", "document", "file", "pdf", "can", "you", "please",
            "give", "explain", "more", "details"
        }
        query_words = set(re.findall(r'\w+', q_lower)) - stop_words

        # Group extracted content by page and section
        sections = []
        seen_texts = set()

        chunks_to_use = context_chunks if is_deep_dive else context_chunks[:4]

        for i, chunk in enumerate(chunks_to_use):
            raw_text = chunk.get("text", "").strip()
            if not raw_text:
                continue

            # Skip Table of Contents / Index chunks unless user explicitly asked for contents
            user_wants_toc = any(w in q_lower for w in ["contents", "table of contents", "index", "chapters list"])
            if not user_wants_toc and self._is_table_of_contents(raw_text):
                continue

            # Clean OCR artifacts and line-broken words
            cleaned = re.sub(r'(\b\w+)[-—–]\s+(\w+\b)', r'\1\2', raw_text)
            cleaned = re.sub(r'[-—–]{2,}', ', ', cleaned)
            cleaned = re.sub(r'[_\[\]]+UDGMENT', 'JUDGMENT:', cleaned, flags=re.IGNORECASE)
            for bad_sym in ['«', '»', '~', '|', '^', '<', '>']:
                cleaned = cleaned.replace(bad_sym, ' ')
            # Repair OCR drop-caps like "C hanakya" -> "Chanakya" or "V ishnugupt" -> "Vishnugupt"
            cleaned = re.sub(r'\b([A-Z])\s+([a-z]{2,})\b', r'\1\2', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()

            meta = chunk.get("metadata", {})
            page = meta.get("page", i + 1)
            src = meta.get("source", "Document")

            # Break into clean sentences
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned) if len(s.strip()) > 20]
            if not sentences:
                continue

            # Take clean sentences in natural reading flow
            take_count = 5 if is_deep_dive else 3
            top_sentences = sentences[:take_count]
            passage = " ".join(top_sentences)
            passage_snippet = passage[:120].lower()
            if passage_snippet not in seen_texts:
                seen_texts.add(passage_snippet)
                sections.append({
                    "src": src,
                    "page": page,
                    "text": passage
                })

        if not sections and context_chunks:
            first_text = re.sub(r'\s+', ' ', context_chunks[0].get("text", "")).strip()
            sections.append({
                "src": context_chunks[0].get("metadata", {}).get("source", "Document"),
                "page": context_chunks[0].get("metadata", {}).get("page", 1),
                "text": first_text[:500]
            })

        # Build clean, elegant response
        response_parts = []
        is_notes = any(w in q_lower for w in ["note", "notes", "takeaway", "takeaways", "summary", "pointers", "bullet"])
        if is_notes:
            response_parts.append("📝 **Key Notes from Document**\n")
        elif is_deep_dive:
            response_parts.append("📖 **Detailed Document Analysis**\n")
        else:
            response_parts.append("📄 **Key Findings from Document**\n")

        for idx, sec in enumerate(sections, 1):
            if len(sections) > 1:
                response_parts.append(f"**Point {idx} (Page {sec['page']}):**\n{sec['text']}\n")
            else:
                response_parts.append(f"*(From Page {sec['page']})*\n{sec['text']}\n")

        if is_deep_dive:
            response_parts.append("💡 *Tip: You can ask about any specific rule, page, or concept for a deeper breakdown.*")
        else:
            response_parts.append("💬 *Say 'explain more' for an expanded deep dive.*")

        response_parts.append("*(Note: Retrieved directly from local on-device ChromaDB vector store in offline mode)*")

        answer_text = "\n".join(response_parts)

        # Collect all referenced pages without 4-page artificial truncation
        all_sources = []
        for sec in sections:
            src_label = f"{sec['src']} (Page {sec['page']})"
            if src_label not in all_sources:
                all_sources.append(src_label)

        return {
            "answer": answer_text,
            "sources": all_sources or sorted(list(sources_set))
        }

    def _is_table_of_contents(self, text: str) -> bool:
        """Detect whether a chunk is merely a structural Table of Contents or Index list."""
        if not text:
            return False
        import re
        t = text.strip()
        if re.search(r'^\s*(?:table\s+of\s+)?contents\b', t, re.IGNORECASE):
            return True
        if re.search(r'^\s*(?:index|subject\s+index|author\s+index)\b', t, re.IGNORECASE):
            return True
        if len(re.findall(r'chapter\s*[-—–:]?\s*\d+', t, re.IGNORECASE)) >= 3:
            return True
        if len(re.findall(r'[\.·\-_]{3,}\s*\d+', t)) >= 3:
            return True
        return False

    def _get_offline_subject_notes(self, q_lower: str, goal: str, skills: str, cap_name: str) -> Optional[str]:
        """Provides rich on-device study notes and quick references for common technical and domain subjects."""
        # 1. Python
        if "python" in q_lower and any(w in q_lower for w in ["note", "notes", "learn", "basics", "summary", "cheat", "study", "what is", "explain"]):
            return (
                f"🐍 **Python Core Study Notes & Quick Reference**\n\n"
                f"### 1. Fundamental Data Types & Structures\n"
                f"• **Primitives:** `int`, `float`, `str`, `bool`\n"
                f"• **Lists:** Ordered, mutable collections: `items = [1, 2, 3]`\n"
                f"• **Tuples:** Immutable sequences: `point = (10, 20)`\n"
                f"• **Dictionaries:** Fast O(1) hash maps: `user = {{'name': '{cap_name}', 'role': 'Engineer'}}`\n"
                f"• **Sets:** Unique unordered elements: `unique_ids = {{1, 2, 3}}`\n\n"
                f"### 2. High-Yield Idiomatic Python\n"
                f"• **List Comprehensions:** `evens = [x for x in range(20) if x % 2 == 0]`\n"
                f"• **Generators (`yield`):** Memory-efficient streaming of large datasets.\n"
                f"• **Context Managers (`with`):** Automatic resource handling (files, DB connections).\n"
                f"• **Decorators (`@`):** Wrappers modifying function behavior (e.g., `@property`, `@dataclass`).\n\n"
                f"### 3. OOP & Clean Architecture\n"
                f"• **Classes & Inheritance:** Encapsulation and polymorphism via `class Child(Parent):`.\n"
                f"• **Dunder Methods:** `__init__`, `__str__`, `__repr__`, `__len__`, `__call__`.\n"
                f"• **Type Annotations:** `def fetch_user(id: int) -> dict:` for type safety.\n\n"
                f"### 4. Essential Ecosystem\n"
                f"• **Web & Microservices:** FastAPI, Uvicorn, Requests.\n"
                f"• **Data & AI:** NumPy, Pandas, PyTorch, ChromaDB.\n"
                f"• **Concurrency:** `asyncio` for non-blocking asynchronous event loops.\n\n"
                f"💡 *Say 'explain decorators in Python' or 'practice a Python interview question' to test your skills!*"
            )

        # 2. Java
        if "java" in q_lower and any(w in q_lower for w in ["note", "notes", "learn", "basics", "summary", "cheat", "study", "what is", "explain"]):
            return (
                f"☕ **Java Core Study Notes & Architecture**\n\n"
                f"### 1. JVM Architecture & Memory\n"
                f"• **JDK vs JRE vs JVM:** JDK is for compilation (`javac`); JVM executes bytecode on the OS.\n"
                f"• **Memory Model:** Heap (objects, GC managed) vs Stack (primitive variables, method execution frames).\n"
                f"• **Garbage Collection:** Generational collection (Eden, Survivor spaces, Tenured).\n\n"
                f"### 2. The 4 Pillars of OOP\n"
                f"• **Encapsulation:** Private variables exposed through public getter/setter methods.\n"
                f"• **Inheritance:** Class hierarchies using `extends` (single inheritance only).\n"
                f"• **Polymorphism:** Overloading (compile-time) vs Overriding with `@Override` (runtime).\n"
                f"• **Abstraction:** Interface contracts (`interface`) vs partial implementations (`abstract class`).\n\n"
                f"### 3. Collections Framework\n"
                f"• **Lists:** `ArrayList` (O(1) index access) vs `LinkedList` (O(1) node insertion).\n"
                f"• **Sets:** `HashSet` (O(1) lookup, hashing) vs `TreeSet` (Red-Black tree sorted).\n"
                f"• **Maps:** `HashMap` (key-value hash table) vs `ConcurrentHashMap` (thread-safe buckets).\n\n"
                f"💡 *Say 'practice a Java mock interview' to start a technical scenario!*"
            )

        # 3. SQL & Databases
        if any(w in q_lower for w in ["sql", "database", "rdbms", "postgres", "mysql"]) and any(w in q_lower for w in ["note", "notes", "learn", "basics", "summary", "cheat", "study"]):
            return (
                f"🗄️ **SQL & Relational Database Study Notes**\n\n"
                f"### 1. Core SQL Commands\n"
                f"• **DDL:** `CREATE`, `ALTER`, `DROP`, `TRUNCATE`\n"
                f"• **DML:** `SELECT`, `INSERT`, `UPDATE`, `DELETE`\n"
                f"• **DCL / TCL:** `GRANT`, `REVOKE`, `COMMIT`, `ROLLBACK`\n\n"
                f"### 2. Essential Joins\n"
                f"• **INNER JOIN:** Matching rows present in both tables.\n"
                f"• **LEFT JOIN:** All rows from left table, matching rows from right.\n"
                f"• **FULL OUTER JOIN:** All rows from both tables, filling `NULL` where no match.\n\n"
                f"### 3. ACID Properties & Transactions\n"
                f"• **Atomicity:** All operations succeed or all roll back.\n"
                f"• **Consistency:** Database transitions only between valid states.\n"
                f"• **Isolation:** Concurrent transactions do not interfere with each other.\n"
                f"• **Durability:** Committed transactions survive system power failures.\n\n"
                f"### 4. Indexing & Optimization\n"
                f"• **B-Tree Indexes:** Accelerate equality (`=`) and range (`BETWEEN`) lookups.\n"
                f"• **Normalization:** 1NF (atomic), 2NF (no partial dependency), 3NF (no transitive dependency).\n\n"
                f"💡 *Say 'give me an SQL interview question' to test your queries!*"
            )

        # 4. Data Structures & Algorithms (DSA)
        if any(w in q_lower for w in ["dsa", "data structure", "data structures", "algorithm", "algorithms"]) and any(w in q_lower for w in ["note", "notes", "summary", "cheat", "study", "basics"]):
            return (
                f"📐 **Data Structures & Algorithms (DSA) Study Notes**\n\n"
                f"### 1. Asymptotic Complexity (Big-O)\n"
                f"• **O(1) Constant:** Hash map lookup, Array index access.\n"
                f"• **O(log N) Logarithmic:** Binary Search.\n"
                f"• **O(N) Linear:** Linear scan, Linked list traversal.\n"
                f"• **O(N log N):** Merge Sort, Quick Sort (average), Heap Sort.\n"
                f"• **O(N²) Quadratic:** Nested loops, Bubble / Selection sort.\n\n"
                f"### 2. Linear Structures\n"
                f"• **Arrays:** Contiguous memory, O(1) random access, O(N) insertion/deletion.\n"
                f"• **Linked Lists:** Dynamic nodes with pointers; O(1) head insertion, O(N) search.\n"
                f"• **Stacks (LIFO) & Queues (FIFO):** Essential for DFS / BFS and expression parsing.\n\n"
                f"### 3. Trees & Graphs\n"
                f"• **Binary Search Tree (BST):** Left child < root < right child; O(log N) average search.\n"
                f"• **Graph Traversal:** BFS (Queue-based, shortest path) vs DFS (Stack/Recursion).\n"
                f"• **Hash Tables:** Collision resolution via Chaining or Open Addressing.\n\n"
                f"💡 *Say 'ask me a DSA interview problem' to practice live!*"
            )

        # 5. Web Development (HTML / CSS / JavaScript)
        if any(w in q_lower for w in ["html", "css", "javascript", "js", "web dev", "frontend"]) and any(w in q_lower for w in ["note", "notes", "learn", "basics", "summary", "cheat", "study"]):
            return (
                f"🌐 **Web Development & Frontend Core Notes**\n\n"
                f"### 1. Semantic HTML5\n"
                f"• Structure content meaningfully: `<header>`, `<nav>`, `<main>`, `<article>`, `<section>`, `<footer>`.\n"
                f"• Accessible forms with `<label for=\"id\">`, `aria-*` tags, and responsive viewport meta.\n\n"
                f"### 2. Modern CSS3 Architecture\n"
                f"• **Flexbox:** 1D layout (`justify-content`, `align-items`, `flex-direction`).\n"
                f"• **CSS Grid:** 2D layout (`grid-template-columns`, `grid-gap`).\n"
                f"• **Responsive Design:** Mobile-first media queries (`@media (max-width: 768px)`).\n\n"
                f"### 3. Modern JavaScript (ES6+)\n"
                f"• **Scope:** `const` and `let` (block-scoped) vs legacy `var` (function-scoped).\n"
                f"• **Asynchronous JS:** Promises and `async/await` handling non-blocking network calls.\n"
                f"• **Event Loop:** Call Stack -> Web APIs -> Microtask Queue (Promises) -> Macrotask Queue (`setTimeout`).\n\n"
                f"💡 *Say 'explain CSS Flexbox vs Grid' or 'practice frontend interview' to dive deeper!*"
            )

        # 6. Biology & Life Sciences
        if any(w in q_lower for w in ["bio", "biology", "botany", "zoology", "genetics", "cell bio", "photosynthesis", "respiration"]) and any(w in q_lower for w in ["note", "notes", "learn", "basics", "summary", "cheat", "study", "what is", "explain"]):
            return (
                f"🔬 **Biology & Life Sciences Core Study Notes**\n\n"
                f"### 1. Cell Biology & Bioenergetics\n"
                f"• **Cell Theory:** All living organisms are composed of cells; the cell is the fundamental unit of life.\n"
                f"• **Prokaryotes vs Eukaryotes:** Prokaryotes lack membrane-bound organelles (e.g. bacteria); Eukaryotes possess a defined nucleus and compartmentalized organelles.\n"
                f"• **Mitochondria & ATP:** Generates cellular energy via Cellular Respiration (Glycolysis in cytoplasm -> Krebs Cycle -> Electron Transport Chain in mitochondria).\n"
                f"• **Chloroplasts & Photosynthesis:** Converts solar energy into glucose ($6CO_2 + 6H_2O \\rightarrow C_6H_{{12}}O_6 + 6O_2$) via Light Reactions and the Calvin Cycle.\n\n"
                f"### 2. Genetics & Molecular Dogma\n"
                f"• **Central Dogma:** DNA Replication -> Transcription ($DNA \\rightarrow mRNA$) -> Translation ($mRNA \\rightarrow Protein$ via Ribosomes).\n"
                f"• **Mendelian Inheritance:** Law of Segregation and Law of Independent Assortment (Dominant vs Recessive alleles).\n"
                f"• **Modern Gene Tech:** PCR DNA amplification, Gel Electrophoresis, and CRISPR-Cas9 targeted genome editing.\n\n"
                f"### 3. Human Physiology & Systems\n"
                f"• **Circulatory:** Double circulation (Systemic & Pulmonary), SA Node natural cardiac pacemaker, Arteries vs Veins.\n"
                f"• **Nervous:** Central (Brain/Spine) vs Peripheral (Sensory/Motor), Action Potentials driven by $Na^+/K^+$ pumps.\n"
                f"• **Endocrine:** Hormonal homeostasis (Insulin/Glucagon blood sugar balance, Adrenaline fight-or-flight).\n\n"
                f"💡 *Tip: Drop any Biology textbook or syllabus PDF into Knowledge Base for instant chapter-level citations!*"
            )

        # 7. Medical & Clinical Sciences
        if any(w in q_lower for w in ["medic", "doctor", "mbbs", "clinical", "anatomy", "neurology", "pharma"]) and any(w in q_lower for w in ["note", "notes", "learn", "basics", "summary", "cheat", "study"]):
            return (
                f"🩺 **Clinical & Medical Science Study Notes**\n\n"
                f"### 1. Clinical Diagnosis & Patient Assessment\n"
                f"• **Vital Signs:** Blood Pressure (<120/80 mmHg), Heart Rate (60–100 bpm), Respiratory Rate (12–20 bpm), SpO2 (>95%).\n"
                f"• **History Taking (SOCRATES):** Site, Onset, Character, Radiation, Associations, Time course, Exacerbating factors, Severity.\n\n"
                f"### 2. Acute Emergency & Pharmacology\n"
                f"• **Acute Stroke Protocol:** CT Non-Contrast to exclude hemorrhage -> Thrombolysis (IV rtPA/Tenecteplase) within 4.5-hour window.\n"
                f"• **Pharmacodynamics & Kinetics:** ADME (Absorption, Distribution, Metabolism via Cytochrome P450, Excretion via kidneys).\n\n"
                f"### 3. Organ Systems & Pathology\n"
                f"• **Cardiovascular:** Myocardial Infarction triage (ST elevation on ECG, Troponin I/T biomarkers).\n"
                f"• **Neurology:** Upper Motor Neuron (hyperreflexia, spasticity) vs Lower Motor Neuron (fasciculations, flaccid paralysis) lesions.\n\n"
                f"💡 *Say 'start clinical mock interview' to practice patient scenario diagnosis!*"
            )

        # 8. Physics & Engineering Fundamentals
        if any(w in q_lower for w in ["physics", "mechanics", "thermo", "optics", "electromagnetism"]) and any(w in q_lower for w in ["note", "notes", "learn", "basics", "summary", "cheat", "study"]):
            return (
                f"⚛️ **Physics & Mechanics Core Study Notes**\n\n"
                f"### 1. Classical Mechanics & Dynamics\n"
                f"• **Newton's Laws:** Inertia ($F_{{net}}=0$), Acceleration ($F=ma$), Action-Reaction ($F_{{AB}}=-F_{{BA}}$).\n"
                f"• **Work-Energy Theorem:** $W = \\Delta KE$; Conservation of Mechanical Energy ($KE_i + PE_i = KE_f + PE_f$).\n\n"
                f"### 2. Electromagnetism & Circuits\n"
                f"• **Maxwell's Equations:** Electric flux, magnetic solenoidal property, Faraday's Law ($EMF = -d\\Phi_B/dt$), Ampère-Maxwell Law.\n"
                f"• **Ohm's Law & Circuit Analysis:** $V = IR$, Kirchhoff's Current Law (KCL) & Voltage Law (KVL).\n\n"
                f"### 3. Thermodynamics\n"
                f"• **Laws of Thermodynamics:** Zeroth (Thermal equilibrium), First ($\\Delta U = Q - W$), Second (Entropy of isolated systems always increases).\n\n"
                f"💡 *Drop any Physics syllabus or engineering PDF into Knowledge Base for deep RAG breakdowns!*"
            )

        # 9. Chemistry & Molecular Sciences
        if any(w in q_lower for w in ["chem", "chemistry", "organic chemistry", "inorganic"]) and any(w in q_lower for w in ["note", "notes", "learn", "basics", "summary", "cheat", "study"]):
            return (
                f"🧪 **Chemistry & Molecular Science Study Notes**\n\n"
                f"### 1. Atomic Structure & Periodic Trends\n"
                f"• **Periodic Trends:** Electronegativity & Ionization Energy increase across period (left -> right), decrease down group.\n"
                f"• **Chemical Bonding:** Ionic (electron transfer), Covalent (electron sharing), Hydrogen bonding & van der Waals forces.\n\n"
                f"### 2. Organic Chemistry & Reaction Mechanisms\n"
                f"• **Nucleophilic Substitution:** $S_N1$ (two-step, carbocation intermediate, racemization) vs $S_N2$ (one-step concerted, backside attack, inversion).\n"
                f"• **Electrophilic Aromatic Substitution:** Benzene resonance stabilization, ortho/para vs meta-directing substituents.\n\n"
                f"### 3. Physical Chemistry & Kinetics\n"
                f"• **Chemical Equilibrium:** Le Chatelier's Principle; Equilibrium constant ($K_{{eq}} = [Products]/[Reactants]$).\n"
                f"• **Reaction Rates:** Arrhenius Equation ($k = A e^{{-E_a/RT}}$) showing temperature dependence of activation energy.\n\n"
                f"💡 *Ask for specific reaction mechanisms or upload chemistry research notes anytime!*"
            )

        # 10. History, Polity & Governance
        if any(w in q_lower for w in ["history", "polity", "constitution", "governance", "upsc", "civil service"]) and any(w in q_lower for w in ["note", "notes", "learn", "basics", "summary", "cheat", "study"]):
            return (
                f"🏛️ **Indian Polity & Constitutional Governance Study Notes**\n\n"
                f"### 1. Constitutional Framework\n"
                f"• **Preamble:** Sovereign, Socialist, Secular, Democratic Republic securing Justice, Liberty, Equality, Fraternity.\n"
                f"• **Fundamental Rights (Part III, Arts 12–35):** Equality (14–18), Freedom (19–22), Against Exploitation (23–24), Freedom of Religion (25–28), Remedies (Art 32 - Heart & Soul of Constitution).\n\n"
                f"### 2. Separation of Powers & Federalism\n"
                f"• **Executive:** President (Art 52), Prime Minister & Council of Ministers (Art 74–75 collective responsibility to Lok Sabha).\n"
                f"• **Legislature:** Bicameral Parliament (Lok Sabha - direct election; Rajya Sabha - permanent federal council).\n"
                f"• **Judiciary:** Integrated independent Supreme Court (Art 124) with Judicial Review (Basic Structure Doctrine - Kesavananda Bharati case).\n\n"
                f"### 3. Public Policy & Administrative Mechanisms\n"
                f"• **Directive Principles (Part IV):** Socio-economic justice guidelines for state governance.\n"
                f"• **NITI Aayog & Decentralization:** Cooperative federalism, 73rd & 74th Constitutional Amendments (Panchayati Raj & Municipalities).\n\n"
                f"💡 *Upload M. Laxmikanth or GS syllabus PDFs to get full chapter-level cross-examinations!*"
            )

        # 11. Universal Field Notes Generator for ANY Academic or Professional Discipline
        import re
        field_match = re.search(r'(?:i need\s+|give me\s+|show me\s+|tell me about\s+)?([a-zA-Z\s]{2,30}?)\s+(?:related\s+)?(?:notes|study notes|cheat sheet|summary)', q_lower)
        if field_match:
            raw_field = field_match.group(1).strip()
            clean_field = re.sub(r'^(?:some|the|a|an|any|my)\s+', '', raw_field, flags=re.IGNORECASE).strip()
            if len(clean_field) >= 3 and clean_field.lower() not in ["profile", "career", "all", "more"]:
                title_field = clean_field.title()
                return (
                    f"📚 **{title_field} Core Study Notes & Academic Framework**\n\n"
                    f"### 1. Foundational Axioms & Scope\n"
                    f"• **Primary Focus:** Systematic study of principles, theories, and empirical models governing **{title_field}**.\n"
                    f"• **Core Objective:** Developing analytical depth, theoretical fluency, and applied problem-solving methodologies in {clean_field}.\n\n"
                    f"### 2. Key Methodological Frameworks\n"
                    f"• **Theoretical Models:** Standard taxonomies, foundational literature, and primary hypotheses defining {clean_field}.\n"
                    f"• **Applied Synthesis:** Translating conceptual rules into practical, observable benchmarks.\n"
                    f"• **Systematic Review:** Identifying variables, boundary conditions, and real-world constraints in {clean_field}.\n\n"
                    f"### 3. High-Yield Revision Strategy for {cap_name}\n"
                    f"• **Milestone 1:** Establish clear definitions of core terminology and primary classification criteria.\n"
                    f"• **Milestone 2:** Solve foundational problems and cross-reference key case studies.\n"
                    f"• **Milestone 3:** Synthesize active recall cards and practice domain-specific scenario discussions.\n\n"
                    f"💡 *Pro-Tip: Upload your {title_field} textbook, PDF syllabus, or lecture slides into the Knowledge Vault for 100% chapter-grounded answers and instant page references!*"
                )

        return None
# -*- coding: utf-8 -*-
"""Everything the bot needs to hold a call in a language other than English.

Nothing else in the project changes shape for this. The rest of the code goes
on writing its canned lines in English and calling _speak() with them; this
module is the one place that knows there is more than one language, and the
three hooks that use it are deliberately small:

  * stt.py      asks for the Deepgram language string and reads the language
                Deepgram detected back out of the response.
  * tts.py      asks which voice and which engine to speak a reply with.
  * bridge.py   passes the caller's language to the model, and localise() runs
                over every canned line on its way into _speak().

WHAT EACH LANGUAGE COSTS (measured against the vendors' own docs, Sept 2026):

  English, Hindi, French, Spanish - free. Deepgram detects all four in one
  stream with `language=multi`, and Kokoro (already local, already paid for)
  has a voice for each. Nothing to switch on.

  Tamil - not free, twice over. Deepgram supports `ta` on nova-3 only, and NOT
  in the multi set, so Tamil cannot be auto-detected: the stream has to be
  reconnected on `language=ta` once we know. And Kokoro has no Tamil voice at
  all - the 54 voices in models/voices.bin cover American and British English,
  Spanish, French, Hindi, Italian, Portuguese, Japanese and Chinese, and that
  is the whole list. Tamil speech therefore goes out through ElevenLabs, which
  does support it (eleven_turbo_v2_5, eleven_flash_v2_5). No ELEVENLABS_API_KEY
  means no Tamil voice, and enable_tamil() below says so rather than the call
  going silent.

The knowledge base stays in English on purpose. all-MiniLM-L6-v2 is an English
embedder, so a Tamil or French question is translated to English for the
retrieval step only (llm.GroqBrain.to_english) and the answer is written back
in the caller's language by the model. Translating the KB itself would mean
maintaining four copies of every answer and re-embedding all of them.
"""
import logging
import re

from config import settings

log = logging.getLogger("jarvis.i18n")

DEFAULT = "en"

# --------------------------------------------------------------------------
# The languages themselves.
#
#   deepgram   language code for a dedicated stream (used when auto-detection
#              cannot cover it, i.e. Tamil)
#   auto       True when Deepgram's `language=multi` can pick it out of the
#              audio with no prompt and no reconnect
#   engine     "kokoro" (local, free) or "elevenlabs" (paid, for Tamil)
#   voice      Kokoro voice id, from models/voices.bin
#   phonemes   the language Kokoro phonemises as - NOT the same string as the
#              Deepgram code, which is why both are here
# --------------------------------------------------------------------------
LANGUAGES = {
    "en": {
        "name": "English", "native": "English",
        "deepgram": settings.deepgram_language or "en-IN", "auto": True,
        "engine": "kokoro", "voice": "af_heart", "phonemes": "en-us",
    },
    "hi": {
        "name": "Hindi", "native": "हिंदी",
        "deepgram": "hi", "auto": True,
        "engine": "kokoro", "voice": "hf_alpha", "phonemes": "hi",
    },
    "fr": {
        "name": "French", "native": "Français",
        "deepgram": "fr", "auto": True,
        "engine": "kokoro", "voice": "ff_siwis", "phonemes": "fr-fr",
    },
    "es": {
        "name": "Spanish", "native": "Español",
        "deepgram": "es", "auto": True,
        "engine": "kokoro", "voice": "ef_dora", "phonemes": "es",
    },
    "ta": {
        "name": "Tamil", "native": "தமிழ்",
        # nova-3 only, and not in the multi set - see the module docstring.
        "deepgram": "ta", "auto": False,
        "engine": "elevenlabs", "voice": None, "phonemes": None,
    },
}

# What a caller says to ask for a language, in any language. Matched on a whole
# word so "Espanola Road" cannot switch a call to Spanish.
REQUESTS = {
    # "தமிழ" with no pulli on purpose - it is the stem every inflected form
    # starts with (தமிழ், தமிழில், தமிழ்ல), and matching the stem catches all
    # of them where matching the citation form catches only one.
    "ta": ["tamil", "தமிழ", "thamizh", "tamizh", "tamil la", "tamil pesunga",
           "speak tamil", "in tamil", "tamil please"],
    "hi": ["hindi", "हिंदी", "हिन्दी", "speak hindi", "in hindi", "hindi mein",
           "hindi please"],
    "fr": ["french", "français", "francais", "en francais", "parlez francais",
           "speak french", "in french"],
    "es": ["spanish", "español", "espanol", "en espanol", "hable espanol",
           "speak spanish", "in spanish"],
    "en": ["english", "in english", "speak english", "english please",
           "back to english"],
}

# --------------------------------------------------------------------------
# The canned lines.
#
# Keyed by the ENGLISH text exactly as bridge.py and llm.py already write it,
# so those modules keep their constants and localise() does the swap on the way
# out. A line that is not in here - anything the model wrote - passes through
# untouched, because the model has already been told to answer in the caller's
# language.
# --------------------------------------------------------------------------
LINES = {
    "greeting_call": {
        "en": "Hello! Thank you for calling Karthipuram. I'm Jarvis. How can I help you today?",
        "hi": "नमस्ते! कार्तिपुरम में कॉल करने के लिए धन्यवाद। मैं जार्विस हूँ। मैं आपकी क्या मदद कर सकता हूँ?",
        "fr": "Bonjour ! Merci d'avoir appelé Karthipuram. Je suis Jarvis. Comment puis-je vous aider aujourd'hui ?",
        "es": "¡Hola! Gracias por llamar a Karthipuram. Soy Jarvis. ¿En qué puedo ayudarle hoy?",
        "ta": "வணக்கம்! கார்த்திபுரத்தைத் தொடர்பு கொண்டதற்கு நன்றி. நான் ஜார்விஸ். இன்று உங்களுக்கு எப்படி உதவ முடியும்?",
    },
    "recording_notice": {
        "en": "This call is recorded for quality purposes.",
        "hi": "गुणवत्ता के लिए यह कॉल रिकॉर्ड की जा रही है।",
        "fr": "Cet appel est enregistré à des fins de qualité.",
        "es": "Esta llamada se graba con fines de calidad.",
        "ta": "தரம் கருதி இந்த அழைப்பு பதிவு செய்யப்படுகிறது.",
    },
    "fallback": {
        "en": "Sorry, I did not catch that. Could you please say it again?",
        "hi": "माफ़ कीजिए, मैं समझ नहीं पाया। क्या आप दोबारा कह सकते हैं?",
        "fr": "Désolé, je n'ai pas compris. Pourriez-vous répéter, s'il vous plaît ?",
        "es": "Disculpe, no le he entendido. ¿Podría repetirlo, por favor?",
        "ta": "மன்னிக்கவும், எனக்குச் சரியாகக் கேட்கவில்லை. மீண்டும் ஒருமுறை சொல்ல முடியுமா?",
    },
    "greeting": {
        "en": "Hello! How can I assist you with the project today?",
        "hi": "नमस्ते! आज मैं इस प्रोजेक्ट के बारे में आपकी क्या मदद कर सकता हूँ?",
        "fr": "Bonjour ! Comment puis-je vous renseigner sur le projet aujourd'hui ?",
        "es": "¡Hola! ¿Cómo puedo ayudarle con el proyecto hoy?",
        "ta": "வணக்கம்! இந்தத் திட்டம் குறித்து உங்களுக்கு எப்படி உதவ முடியும்?",
    },
    "unclear": {
        "en": "I'm sorry, I didn't catch that clearly. Could you please repeat?",
        "hi": "माफ़ कीजिए, मुझे साफ़ सुनाई नहीं दिया। कृपया दोहराएँ।",
        "fr": "Désolé, je n'ai pas bien entendu. Pouvez-vous répéter, s'il vous plaît ?",
        "es": "Lo siento, no le he oído con claridad. ¿Puede repetirlo, por favor?",
        "ta": "மன்னிக்கவும், அது தெளிவாகக் கேட்கவில்லை. தயவுசெய்து மீண்டும் சொல்லுங்கள்.",
    },
    "off_topic": {
        "en": "I can help you with details about our project. What would you like to know?",
        "hi": "मैं आपको हमारे प्रोजेक्ट की जानकारी दे सकता हूँ। आप क्या जानना चाहेंगे?",
        "fr": "Je peux vous renseigner sur notre projet. Que souhaitez-vous savoir ?",
        "es": "Puedo darle información sobre nuestro proyecto. ¿Qué le gustaría saber?",
        "ta": "எங்கள் திட்டம் பற்றிய விவரங்களில் நான் உதவ முடியும். என்ன தெரிந்து கொள்ள விரும்புகிறீர்கள்?",
    },
    "vague": {
        "en": "Could you please tell me what details you're looking for regarding the project?",
        "hi": "क्या आप बता सकते हैं कि प्रोजेक्ट के बारे में आपको कौन सी जानकारी चाहिए?",
        "fr": "Pourriez-vous me dire quelles informations vous cherchez sur le projet ?",
        "es": "¿Podría decirme qué información busca sobre el proyecto?",
        "ta": "திட்டம் குறித்து என்ன விவரங்கள் வேண்டும் என்று சொல்ல முடியுமா?",
    },
    "not_found": {
        "en": "I'm sorry, I don't have that specific information right now. Would you like me to arrange a call with our sales team?",
        "hi": "माफ़ कीजिए, वह जानकारी अभी मेरे पास नहीं है। क्या मैं हमारी सेल्स टीम से कॉल की व्यवस्था करूँ?",
        "fr": "Désolé, je n'ai pas cette information précise pour le moment. Souhaitez-vous que j'organise un appel avec notre équipe commerciale ?",
        "es": "Lo siento, no tengo esa información concreta ahora mismo. ¿Desea que organice una llamada con nuestro equipo de ventas?",
        "ta": "மன்னிக்கவும், அந்தக் குறிப்பிட்ட தகவல் இப்போது என்னிடம் இல்லை. எங்கள் விற்பனைக் குழுவிடம் ஒரு அழைப்பை ஏற்பாடு செய்யட்டுமா?",
    },
    "cta": {
        "en": "Would you like me to schedule a call with our sales team or send you the brochure on WhatsApp?",
        "hi": "क्या मैं सेल्स टीम से कॉल तय करूँ, या आपको व्हाट्सएप पर ब्रोशर भेज दूँ?",
        "fr": "Souhaitez-vous que je planifie un appel avec notre équipe commerciale, ou que je vous envoie la brochure sur WhatsApp ?",
        "es": "¿Desea que programe una llamada con nuestro equipo de ventas o que le envíe el folleto por WhatsApp?",
        "ta": "எங்கள் விற்பனைக் குழுவுடன் ஒரு அழைப்பை ஏற்பாடு செய்யவா, அல்லது வாட்ஸ்அப்பில் ப்ரோஷர் அனுப்பவா?",
    },
    "exit": {
        "en": "Thank you for your time. I'll share the brochure with you on WhatsApp. Have a great day!",
        "hi": "आपके समय के लिए धन्यवाद। ब्रोशर मैं आपको व्हाट्सएप पर भेज दूँगा। आपका दिन शुभ हो!",
        "fr": "Merci pour votre temps. Je vous envoie la brochure sur WhatsApp. Bonne journée !",
        "es": "Gracias por su tiempo. Le enviaré el folleto por WhatsApp. ¡Que tenga un buen día!",
        "ta": "உங்கள் நேரத்திற்கு நன்றி. ப்ரோஷரை வாட்ஸ்அப்பில் அனுப்புகிறேன். நல்ல நாளாக அமையட்டும்!",
    },
    "handoff_ask_time": {
        "en": "Of course. I'll arrange for one of our sales specialists to call you back. Which day and what time would suit you?",
        "hi": "ज़रूर। हमारा एक सेल्स विशेषज्ञ आपको वापस कॉल करेगा। कौन सा दिन और कौन सा समय आपको ठीक रहेगा?",
        "fr": "Bien sûr. Je vais faire en sorte qu'un de nos conseillers vous rappelle. Quel jour et à quelle heure vous conviendrait ?",
        "es": "Por supuesto. Haré que uno de nuestros asesores le llame. ¿Qué día y a qué hora le vendría bien?",
        "ta": "கண்டிப்பாக. எங்கள் விற்பனை நிபுணர் ஒருவர் உங்களைத் திரும்ப அழைப்பார். எந்த நாள், எத்தனை மணிக்கு வசதியாக இருக்கும்?",
    },
    "handoff_confirmed": {
        "en": "Perfect. Our specialist will call you then, and I'll send you a confirmation on WhatsApp and SMS right away.",
        "hi": "बढ़िया। हमारा विशेषज्ञ आपको उसी समय कॉल करेगा, और मैं अभी व्हाट्सएप और एसएमएस पर पुष्टि भेज देता हूँ।",
        "fr": "Parfait. Notre conseiller vous appellera à ce moment-là, et je vous envoie tout de suite une confirmation par WhatsApp et SMS.",
        "es": "Perfecto. Nuestro asesor le llamará entonces, y le envío ahora mismo una confirmación por WhatsApp y SMS.",
        "ta": "சரி. எங்கள் நிபுணர் அப்போது உங்களை அழைப்பார். உறுதிப்படுத்தலை வாட்ஸ்அப் மற்றும் எஸ்எம்எஸ் மூலம் இப்போதே அனுப்புகிறேன்.",
    },
    "handoff_declined": {
        "en": "No problem. Is there anything else I can help you with?",
        "hi": "कोई बात नहीं। क्या मैं और किसी चीज़ में आपकी मदद कर सकता हूँ?",
        "fr": "Pas de problème. Puis-je vous aider avec autre chose ?",
        "es": "No hay problema. ¿Puedo ayudarle con algo más?",
        "ta": "பரவாயில்லை. வேறு ஏதேனும் உதவி வேண்டுமா?",
    },
    "handoff_ask_hour": {
        "en": "Sure. What time exactly - say, ten in the morning, or six in the evening?",
        "hi": "ठीक है। ठीक कितने बजे - जैसे सुबह दस, या शाम छह?",
        "fr": "Très bien. À quelle heure exactement - dix heures du matin, ou six heures du soir ?",
        "es": "De acuerdo. ¿A qué hora exactamente: las diez de la mañana, o las seis de la tarde?",
        "ta": "சரி. சரியாக எத்தனை மணிக்கு - காலை பத்து மணியா, மாலை ஆறு மணியா?",
    },
    "schedule_ask_when": {
        "en": "Yes, of course - I can arrange that for you. Which day and what time would suit you?",
        "hi": "जी हाँ, मैं यह व्यवस्था कर देता हूँ। कौन सा दिन और कौन सा समय आपको ठीक रहेगा?",
        "fr": "Oui, bien sûr, je peux organiser cela. Quel jour et à quelle heure vous conviendrait ?",
        "es": "Sí, claro, puedo organizarlo. ¿Qué día y a qué hora le vendría bien?",
        "ta": "ஆம், கண்டிப்பாக ஏற்பாடு செய்கிறேன். எந்த நாள், எத்தனை மணிக்கு வசதியாக இருக்கும்?",
    },
    "ask_name": {
        "en": "Thank you. And may I have your name, please?",
        "hi": "धन्यवाद। और क्या मैं आपका नाम जान सकता हूँ?",
        "fr": "Merci. Et puis-je avoir votre nom, s'il vous plaît ?",
        "es": "Gracias. ¿Me puede decir su nombre, por favor?",
        "ta": "நன்றி. உங்கள் பெயரைச் சொல்ல முடியுமா?",
    },
    "confirm_name": {
        "en": "Thank you. I have that as {name}. Is that right?",
        "hi": "धन्यवाद। मैंने {name} लिखा है। क्या यह सही है?",
        "fr": "Merci. J'ai noté {name}. Est-ce correct ?",
        "es": "Gracias. He anotado {name}. ¿Es correcto?",
        "ta": "நன்றி. {name} என்று குறித்துக் கொண்டேன். சரிதானா?",
    },
    "ask_name_again": {
        "en": "Sorry, I didn't catch that. Could you say just your name, slowly?",
        "hi": "माफ़ कीजिए, मैं समझ नहीं पाया। क्या आप सिर्फ़ अपना नाम धीरे से बोल सकते हैं?",
        "fr": "Désolé, je n'ai pas saisi. Pouvez-vous dire juste votre nom, lentement ?",
        "es": "Disculpe, no lo he captado. ¿Puede decir solo su nombre, despacio?",
        "ta": "மன்னிக்கவும், கேட்கவில்லை. உங்கள் பெயரை மட்டும் மெதுவாகச் சொல்ல முடியுமா?",
    },
    "closing_offer": {
        "en": "One last thing before you go - would you like me to arrange a call back from one of our sales specialists?",
        "hi": "जाने से पहले एक आख़िरी बात - क्या मैं हमारे किसी सेल्स विशेषज्ञ से कॉल बैक की व्यवस्था करूँ?",
        "fr": "Une dernière chose avant de raccrocher - souhaitez-vous qu'un de nos conseillers vous rappelle ?",
        "es": "Una última cosa antes de colgar: ¿desea que uno de nuestros asesores le devuelva la llamada?",
        "ta": "கடைசியாக ஒன்று - எங்கள் விற்பனை நிபுணர் ஒருவர் உங்களைத் திரும்ப அழைக்க ஏற்பாடு செய்யவா?",
    },
    "switched": {
        "en": "Of course, I'll continue in English.",
        "hi": "ज़रूर, मैं हिंदी में बात करता हूँ।",
        "fr": "Bien sûr, je continue en français.",
        "es": "Por supuesto, continúo en español.",
        "ta": "கண்டிப்பாக, தமிழில் தொடர்கிறேன்.",
    },
    "no_tamil_voice": {
        "en": "I can understand Tamil, but I can only speak English on this line. I'll carry on in English.",
        "ta": "I can understand Tamil, but I can only speak English on this line. I'll carry on in English.",
    },
}

# English text -> key, built once. This is what lets every other module keep
# writing plain English constants.
_BY_ENGLISH = {}
_TEMPLATES = []
for _key, _variants in LINES.items():
    _en = _variants.get("en")
    if not _en:
        continue
    if "{name}" in _en:
        _TEMPLATES.append(
            (_key, re.compile("^" + re.escape(_en).replace(r"\{name\}",
                                                           "(?P<name>.+?)") + "$")))
    else:
        _BY_ENGLISH[_en] = _key


# ------------------------------------------------------------------ helpers

def enabled():
    """The language codes this deployment will actually use."""
    raw = (settings.supported_languages or DEFAULT)
    codes = [c.strip().lower() for c in raw.split(",") if c.strip()]
    out = [DEFAULT] + [c for c in codes if c in LANGUAGES and c != DEFAULT]
    return list(dict.fromkeys(out))


def is_enabled(code) -> bool:
    return code in enabled()


def get(code):
    return LANGUAGES.get(code) or LANGUAGES[DEFAULT]


def name(code) -> str:
    return get(code)["name"]


def auto_detected():
    """Codes Deepgram's multi stream can pick out on its own."""
    return [c for c in enabled() if LANGUAGES[c]["auto"]]


def needs_dedicated_stream(code) -> bool:
    """True for a language that cannot ride the multi stream - Tamil."""
    return is_enabled(code) and not get(code)["auto"]


def tamil_voice_available() -> bool:
    return bool(settings.elevenlabs_api_key)


def speakable(code) -> str:
    """The language the bot can actually SPEAK for a caller using `code`.

    Tamil without an ElevenLabs key is understood but cannot be spoken, and
    saying so beats a silent call.
    """
    if not is_enabled(code):
        return DEFAULT
    if get(code)["engine"] == "elevenlabs" and not tamil_voice_available():
        log.warning("No ELEVENLABS_API_KEY, so %s cannot be spoken - replying "
                    "in English.", name(code))
        return DEFAULT
    return code


def requested(text: str):
    """The language a caller is ASKING for, or None.

    Deepgram's multi stream does not carry Tamil, so this is the only way into
    it - and it is also how a caller gets back out of a language the detector
    put them in by mistake.
    """
    if not text:
        return None
    # ASCII punctuation only. \w drops Tamil and Devanagari combining
    # marks - the pulli in "தமிழ்" is not alphanumeric to Python - and a
    # trigger with its vowel signs filed off matches nothing.
    low = re.sub(r"[!-/:-@\[-`{-~]+", " ", str(text).lower())
    low = " " + re.sub(r"\s+", " ", low).strip() + " "
    for code in enabled():
        for phrase in REQUESTS.get(code, []):
            # An English trigger has to be a whole word, or "Espanola Road"
            # switches the call to Spanish. A trigger in its own script does
            # not: Tamil agglutinates, so a caller asks in "தமிழ்ல", which is
            # "தமிழ்" with a case ending welded on and would never match as a
            # standalone word.
            if phrase.isascii():
                if " %s " % phrase in low:
                    return code
            elif phrase in low:
                return code
    return None


def localize(text: str, code: str) -> str:
    """Swap one canned English line for its translation.

    Anything the model wrote is already in the caller's language and is not in
    the table, so it passes through untouched.
    """
    if not text or code == DEFAULT or not is_enabled(code):
        return text

    stripped = text.strip()
    key = _BY_ENGLISH.get(stripped)
    if key:
        return LINES[key].get(code) or text

    for key, pattern in _TEMPLATES:
        m = pattern.match(stripped)
        if m:
            translated = LINES[key].get(code)
            if translated:
                return translated.format(**m.groupdict())
    return text


def line(key: str, code: str = DEFAULT) -> str:
    """A canned line by key, falling back to English."""
    variants = LINES.get(key) or {}
    return variants.get(code) or variants.get(DEFAULT) or ""

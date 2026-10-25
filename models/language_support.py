"""Documented language metadata used to keep model choices explicit and local."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    scheduled: bool = True


SCHEDULED_LANGUAGES = (
    Language("as", "Assamese"),
    Language("bn", "Bengali"),
    Language("brx", "Bodo"),
    Language("doi", "Dogri"),
    Language("gu", "Gujarati"),
    Language("hi", "Hindi"),
    Language("kn", "Kannada"),
    Language("ks", "Kashmiri"),
    Language("kok", "Konkani"),
    Language("mai", "Maithili"),
    Language("ml", "Malayalam"),
    Language("mni", "Manipuri"),
    Language("mr", "Marathi"),
    Language("ne", "Nepali"),
    Language("or", "Odia"),
    Language("pa", "Punjabi"),
    Language("sa", "Sanskrit"),
    Language("sat", "Santali"),
    Language("sd", "Sindhi"),
    Language("ta", "Tamil"),
    Language("te", "Telugu"),
    Language("ur", "Urdu"),
)

# These are explicitly listed in the adapter metadata rather than inferred
# from a language code.  A model must opt in before the UI exposes a language.
ENGLISH = Language("en", "English", scheduled=False)
WHISPER_ADDITIONAL_LANGUAGES = (
    ENGLISH,
    Language("ar", "Arabic", scheduled=False),
    Language("zh", "Chinese", scheduled=False),
    Language("fr", "French", scheduled=False),
    Language("de", "German", scheduled=False),
    Language("id", "Indonesian", scheduled=False),
    Language("it", "Italian", scheduled=False),
    Language("ja", "Japanese", scheduled=False),
    Language("ko", "Korean", scheduled=False),
    Language("pt", "Portuguese", scheduled=False),
    Language("ru", "Russian", scheduled=False),
    Language("es", "Spanish", scheduled=False),
    Language("th", "Thai", scheduled=False),
    Language("tr", "Turkish", scheduled=False),
    Language("vi", "Vietnamese", scheduled=False),
)


def language_map(codes: set[str]) -> dict[str, Language]:
    all_languages = SCHEDULED_LANGUAGES + WHISPER_ADDITIONAL_LANGUAGES
    return {item.code: item for item in all_languages if item.code in codes}

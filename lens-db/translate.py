import requests

LIBRETRANSLATE_URL = "http://localhost:5000/translate"


def translate(text, source="en", target="zh"):
    response = requests.post(
        LIBRETRANSLATE_URL,
        data={
            "q": text,
            "source": source,
            "target": target,
        },
    )
    response.raise_for_status()
    return response.json()["translatedText"]


if __name__ == "__main__":
    print(
        translate(
            "This is a test of the translation function. It should translate this text from English to whatever language is specified in the function definition."
        )
    )

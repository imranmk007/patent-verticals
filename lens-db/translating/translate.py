import argostranslate.translate


def translate(text, source, target):
    return argostranslate.translate.translate(text, source, target)

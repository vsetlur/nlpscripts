### Main script
# Language detection (some responses are in Spanish)
# Translate above where possible.
# Contraction cleanup / expansion (can't becomes can not)
# Sentiment scoring
# Split out sentences & number
# Label Parts of speech (ideally the tag, but join back in the explanations)
# Label prepositions and (if possible) identify all prepositional phrases)
# Break out each word into its own row and number by order in sentence
# If possible, label emotions.

import fasttext
from pycountry import languages
from googletrans import Translator

def detect_lang(sentences):
    PRETRAINED_MODEL_PATH = 'data/lid.176.bin'
    model = fasttext.load_model(PRETRAINED_MODEL_PATH)

    for sentence in sentences:
        predictions = model.predict(sentence)
        print(predictions)
        if len(predictions) > 0:
            lang_code = predictions[0][0][0].split('__label__')[1]
            print(lang_code)
            lang_name = languages.get(alpha_2=lang_code).name
            print(lang_name)
            # Translate the sentence if it is not English
            if (lang_name != 'English'):
                translate(sentence)


def translate(sentence):
    translator = Translator()
    translations = translator.translate(sentence, dest='en')
    for translation in translations:
        print(translation.origin, ' -> ', translation.text)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    detect_lang([['je mange de la nourriture'], ['Hoy es un buen día!'], ['the red fox sat in the moonlight.']])



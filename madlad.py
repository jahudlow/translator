import json
import modal
import os

from pydantic import BaseModel
from typing import List



# Model and tokenizer to pull from HF.
hfmodel = "jbochi/madlad400-3b-mt"
cache_path = "/vol/cache"

# Dictionary for converting from iso code to bcp-47 code
lang_dict = {
    'eng': 'en',
    'rus': 'ru',
    'spa': 'es',
    'fra': 'fr',
    'deu': 'de',
    'ita': 'it',
    'por': 'pt',
    'pol': 'pl',
    'nld': 'nl',
    'vie': 'vi',
    'tur': 'tr',
    'swe': 'sv',
    'ind': 'id',
    'ron': 'ro',
    'ces': 'cs',
    'zho': 'zh',
    'hun': 'hu',
    'jpn': 'ja',
    'tha': 'th',
    'fin': 'fi',
    'fas': 'fa',
    'ukr': 'uk',
    'dan': 'da',
    'ell': 'el',
    'nor': 'no',
    'bul': 'bg',
    'slk': 'sk',
    'kor': 'ko',
    'ara': 'ar',
    'lit': 'lt',
    'cat': 'ca',
    'slv': 'sl',
    'heb': 'he',
    'est': 'et',
    'lav': 'lv',
    'hin': 'hi',
    'sqi': 'sq',
    'msa': 'ms',
    'aze': 'az',
    'srp': 'sr',
    'tam': 'ta',
    'hrv': 'hr',
    'kaz': 'kk',
    'isl': 'is',
    'mal': 'ml',
    'mar': 'mr',
    'tel': 'te',
    'afr': 'af',
    'glg': 'gl',
    'bel': 'be',
    'mkd': 'mk',
    'eus': 'eu',
    'ben': 'bn',
    'kat': 'ka',
    'mon': 'mn',
    'bos': 'bs',
    'uzb': 'uz',
    'urd': 'ur',
    'swa': 'sw',
    'nep': 'ne',
    'kan': 'kn',
    'guj': 'gu',
    'sin': 'si',
    'cym': 'cy',
    'epo': 'eo',
    'lat': 'la',
    'hye': 'hy',
    'kir': 'ky',
    'tgk': 'tg',
    'gle': 'ga',
    'mlt': 'mt',
    'mya': 'my',
    'khm': 'km',
    'tat': 'tt',
    'som': 'so',
    'kur': 'ku',
    'pus': 'ps',
    'pan': 'pa',
    'kin': 'rw',
    'lao': 'lo',
    'hau': 'ha',
    'div': 'dv',
    'fry': 'fy',
    'ltz': 'lb',
    'mlg': 'mg',
    'gla': 'gd',
    'amh': 'am',
    'uig': 'ug',
    'hat': 'ht',
    'snd': 'sd',
    'jav': 'jv',
    'mri': 'mi',
    'tuk': 'tk',
    'yid': 'yi',
    'bak': 'ba',
    'fao': 'fo',
    'ori': 'or',
    'xho': 'xh',
    'sun': 'su',
    'kal': 'kl',
    'nya': 'ny',
    'smo': 'sm',
    'sna': 'sn',
    'cos': 'co',
    'zul': 'zu',
    'ibo': 'ig',
    'yor': 'yo',
    'sot': 'st',
    'asm': 'as',
    'oci': 'oc',
    'chv': 'cv',
    'bre': 'br',
    'roh': 'rm',
    'san': 'sa',
    'bod': 'bo',
    'orm': 'om',
    'sme': 'se',
    'che': 'ce',
    'oss': 'os',
    'lug': 'lg',
    'tir': 'ti',
    'tso': 'ts',
    'ewe': 'ee',
    'ava': 'av',
    'ton': 'to',
    'tsn': 'tn',
    'fij': 'fj',
    'aka': 'ak',
    'dzo': 'dz',
    'lin': 'ln',
    'grn': 'gn',
    'wln': 'wa',
    'sag': 'sg',
    'lub': 'lu',
    'aym': 'ay',
    'que': 'qu',
    'zha': 'za',
    'ven': 've',
    'nav': 'nv',
    'kom': 'kv',
    'iku': 'iu',
    'hmo': 'ho',
    'cor': 'kw',
    'glv': 'gv',
    'kua': 'kj',
    'ssw': 'ss',
    'wol': 'wo',
    'bam': 'bm',
    'kon': 'kg',
    'cha': 'ch',
    'mah': 'mh',
    'bis': 'bi',
    'nbl': 'nr',
    'run': 'rn',
    'oji': 'oj',
    'kas': 'ks',
    'ful': 'ff',
    'aar': 'aa'
    }


# Download the model.
def download_model():
    import transformers
    model = transformers.T5ForConditionalGeneration.from_pretrained(
        hfmodel,
        cache_dir=cache_path
    )
    model.save_pretrained(cache_path)
    tokenizer = transformers.T5Tokenizer.from_pretrained(
        hfmodel,
        cache_dir=cache_path
    )
    tokenizer.save_pretrained(cache_path)


# Manage suffix on modal endpoint if testing.
suffix = ''
if os.environ.get('MODAL_TEST') == 'TRUE':
    suffix = '_test'


# Define the modal stub.
stub = modal.Stub(
    "madlad400-3b-mt" + suffix,
    image=modal.Image.debian_slim().pip_install(
        "torch", 
        "transformers==4.30.1", 
        "accelerate==0.21.0", 
        "huggingface_hub==0.19.4", 
        "hf-transfer==0.1.4", 
        "sentencepiece",
    ).run_function(
        download_model,
    ),
)


# Define the model.
@stub.cls(gpu='any', secrets=[modal.Secret.from_name("aqua-aws")])
class Model:
    def __enter__(self):
        import transformers
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            hfmodel, cache_dir=cache_path)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            hfmodel, cache_dir=cache_path)

    @modal.method()
    def predict(self, input: dict, 
                #bucket: str, region: str
                ) -> str:
        text = input['text']
        try:
            target_lang = lang_dict[input['target']]
        except KeyError:
            target_lang = input['target']
        text = f'<2{target_lang}> ' + text
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)
        outputs = self.model.generate(input_ids=input_ids, max_length=140)
        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return prediction


# # The predict logic and output formatting.
# @stub.function()
# def predict(input: Input) -> Output:

#     # Import the model.
#     model = Model()

#     # Run the inference.
#     prediction = model.predict.call(input)

#     return prediction


if __name__ == "__main__":
    stub.serve()

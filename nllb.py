from pydantic import BaseModel
import os
from typing import List

import modal


# Model and tokenizer to pull from HF.
hfmodel = "facebook/nllb-200-distilled-1.3B"
cache_path = "/vol/cache"

# Supported languages.
LANGS = {
    "ace": "ace_Arab",
    "acm": "acm_Arab",
    "acq": "acq_Arab",
    "aeb": "aeb_Arab",
    "afr": "afr_Latn",
    "ajp": "ajp_Arab",
    "aka": "aka_Latn",
    "amh": "amh_Ethi",
    "apc": "apc_Arab",
    "arb": "arb_Arab",
    "ars": "ars_Arab",
    "ary": "ary_Arab",
    "arz": "arz_Arab",
    "asm": "asm_Beng",
    "ast": "ast_Latn",
    "awa": "awa_Deva",
    "ayr": "ayr_Latn",
    "azb": "azb_Arab",
    "azj": "azj_Latn",
    "bak": "bak_Cyrl",
    "bam": "bam_Latn",
    "ban": "ban_Latn",
    "bel": "bel_Cyrl",
    "bem": "bem_Latn",
    "ben": "ben_Beng",
    "bho": "bho_Deva",
    "bjn": "bjn_Arab",
    "bod": "bod_Tibt",
    "bos": "bos_Latn",
    "bug": "bug_Latn",
    "bul": "bul_Cyrl",
    "cat": "cat_Latn",
    "ceb": "ceb_Latn",
    "ces": "ces_Latn",
    "cjk": "cjk_Latn",
    "ckb": "ckb_Arab",
    "crh": "crh_Latn",
    "cym": "cym_Latn",
    "dan": "dan_Latn",
    "deu": "deu_Latn",
    "dik": "dik_Latn",
    "dyu": "dyu_Latn",
    "dzo": "dzo_Tibt",
    "ell": "ell_Grek",
    "eng": "eng_Latn",
    "epo": "epo_Latn",
    "est": "est_Latn",
    "eus": "eus_Latn",
    "ewe": "ewe_Latn",
    "fao": "fao_Latn",
    "pes": "pes_Arab",
    "fij": "fij_Latn",
    "fin": "fin_Latn",
    "fon": "fon_Latn",
    "fra": "fra_Latn",
    "fur": "fur_Latn",
    "fuv": "fuv_Latn",
    "gla": "gla_Latn",
    "gle": "gle_Latn",
    "glg": "glg_Latn",
    "grn": "grn_Latn",
    "guj": "guj_Gujr",
    "hat": "hat_Latn",
    "hau": "hau_Latn",
    "heb": "heb_Hebr",
    "hin": "hin_Deva",
    "hne": "hne_Deva",
    "hry": "hrv_Latn",
    "hun": "hun_Latn",
    "hye": "hye_Armn",
    "ibo": "ibo_Latn",
    "ilo": "ilo_Latn",
    "ind": "ind_Latn",
    "isl": "isl_Latn",
    "ita": "ita_Latn",
    "jav": "jav_Latn",
    "jpn": "jpn_Jpan",
    "kab": "kab_Latn",
    "kac": "kac_Latn",
    "kam": "kam_Latn",
    "kan": "kan_Knda",
    "kas": "kas_Arab",
    "kat": "kat_Geor",
    "knc": "knc_Arab",
    "kaz": "kaz_Cyrl",
    "kbp": "kbp_Latn",
    "kea": "kea_Latn",
    "khm": "khm_Khmr",
    "kik": "kik_Latn",
    "kin": "kin_Latn",
    "kir": "kir_Cyrl",
    "kmb": "kmb_Latn",
    "kon": "kon_Latn",
    "kor": "kor_Hang",
    "kmr": "kmr_Latn",
    "lao": "lao_Laoo",
    "lvs": "lvs_Latn",
    "lij": "lij_Latn",
    "lim": "lim_Latn",
    "lin": "lin_Latn",
    "lit": "lit_Latn",
    "lmo": "lmo_Latn",
    "ltg": "ltg_Latn",
    "ltz": "ltz_Latn",
    "lua": "lua_Latn",
    "lug": "lug_Latn",
    "luo": "luo_Latn",
    "lus": "lus_Latn",
    "mag": "mag_Deva",
    "mai": "mai_Deva",
    "mal": "mal_Mlym",
    "mar": "mar_Deva",
    "min": "min_Latn",
    "mkd": "mkd_Cyrl",
    "plt": "plt_Latn",
    "mlt": "mlt_Latn",
    "mni": "mni_Beng",
    "khk": "khk_Cyrl",
    "mos": "mos_Latn",
    "mri": "mri_Latn",
    "zsm": "zsm_Latn",
    "mya": "mya_Mymr",
    "nld": "nld_Latn",
    "nno": "nno_Latn",
    "nob": "nob_Latn",
    "npi": "npi_Deva",
    "nso": "nso_Latn",
    "nus": "nus_Latn",
    "nya": "nya_Latn",
    "oci": "oci_Latn",
    "gaz": "gaz_Latn",
    "ory": "ory_Orya",
    "pag": "pag_Latn",
    "pan": "pan_Guru",
    "pap": "pap_Latn",
    "pol": "pol_Latn",
    "por": "por_Latn",
    "prs": "prs_Arab",
    "pbt": "pbt_Arab",
    "quy": "quy_Latn",
    "ron": "ron_Latn",
    "run": "run_Latn",
    "rus": "rus_Cyrl",
    "sag": "sag_Latn",
    "san": "san_Deva",
    "sat": "sat_Beng",
    "scn": "scn_Latn",
    "shn": "shn_Mymr",
    "sin": "sin_Sinh",
    "slk": "slk_Latn",
    "slv": "slv_Latn",
    "smo": "smo_Latn",
    "sna": "sna_Latn",
    "snd": "snd_Arab",
    "som": "som_Latn",
    "sot": "sot_Latn",
    "spa": "spa_Latn",
    "als": "als_Latn",
    "srd": "srd_Latn",
    "srp": "srp_Cyrl",
    "ssw": "ssw_Latn",
    "sun": "sun_Latn",
    "swe": "swe_Latn",
    "swh": "swh_Latn",
    "szl": "szl_Latn",
    "tam": "tam_Taml",
    "tat": "tat_Cyrl",
    "tel": "tel_Telu",
    "tgk": "tgk_Cyrl",
    "tgl": "tgl_Latn",
    "tha": "tha_Thai",
    "tir": "tir_Ethi",
    "taq": "taq_Latn",
    "tpi": "tpi_Latn",
    "tsn": "tsn_Latn",
    "tso": "tso_Latn",
    "tuk": "tuk_Latn",
    "tum": "tum_Latn",
    "tur": "tur_Latn",
    "twi": "twi_Latn",
    "tzm": "tzm_Tfng",
    "uig": "uig_Arab",
    "ukr": "ukr_Cyrl",
    "umb": "umb_Latn",
    "urd": "urd_Arab",
    "uzn": "uzn_Latn",
    "vec": "vec_Latn",
    "vie": "vie_Latn",
    "war": "war_Latn",
    "wol": "wol_Latn",
    "xho": "xho_Latn",
    "ydd": "ydd_Hebr",
    "yor": "yor_Latn",
    "yue": "yue_Hant",
    "cmn": "zho_Hans",
    "zul": "zul_Latn"
}


# Download the model.
def download_model():
    import transformers
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        hfmodel,
        cache_dir=cache_path
    )
    model.save_pretrained(cache_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
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
    "mt_nllb-distilled" + suffix,
    image=modal.Image.debian_slim().pip_install(
        'torch',
        'transformers',
    ).run_function(
        download_model,
    ),
)


# Define the model.
class Model:
    def __enter__(self):
        import transformers
        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            hfmodel, cache_dir=cache_path)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            hfmodel, cache_dir=cache_path)

    @stub.function(cpu=8, retries=3)
    def predict(self, input: dict) -> str:
        from transformers import pipeline
        pipe = pipeline(
            'translation',
            model=self.model,
            tokenizer=self.tokenizer,
            src_lang=LANGS[input['source']],
            tgt_lang=LANGS[input['target']],
            max_length=512)
        prediction = pipe(input['text'])[0]['translation_text']
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
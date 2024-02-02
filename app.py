from pydantic import BaseModel
from io import StringIO
import psycopg2
import os
import time

import streamlit as st
import pandas as pd
import modal 

#st.set_page_config(layout="wide")

# Streamlit setup
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.title("SIL AI & NLP Translation App 🚀")

# Connect to database using env var
conn = psycopg2.connect(os.getenv("DATABASE_URL"))

# NLLB languages
# Supported languages.
LANGS = {
    "ace": "ace_Arab",
    "acm": "acm_Arab",
    "acq": "acq_Arab",
    "aeb": "aeb_Arab",
    "afr": "afr_Latn",
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

MODELS = {
    "NLLB (distilled)": "mt_nllb-distilled",
    "MADLAD-400": "madlad400-3b-mt"
}

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data
def process_file(uploaded_file):
    if isinstance(uploaded_file, str):
        with open(uploaded_file, 'rb') as f:
            lines = [line.strip() for line in f.readlines()]
    else:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        string_data = stringio.read()
        lines = [line.strip() for line in string_data.split('\n')]
    df = pd.DataFrame(lines, columns=['text'])
    return df

@st.cache_data
def translate(inputs, model):
    f = modal.Function.lookup(MODELS[model], "Model.predict")
    translations = list(f.map(inputs))
    return translations

@st.cache_data
def score(inputs):
    f = modal.Function.lookup("qe-comet", "Model.predict")
    scores = list(f.map(inputs))
    return scores

class Input(BaseModel):
    source: str
    target: str
    text: str

# Upload file to translate.
st.header('Instructions')
st.markdown("""This app translates text files (*.txt) from one language to another. To translate a file:

1. Create a *.txt file with the text you want to translate. This is contain 1-2 sentences per line:

   ```
   This is a first thing I want to translate.
   A second thing I want to translate.
   Here's one more thing I want to translate.
   etc...
   ```

2. Click the "Choose a *.txt file to translate:" button below and select your file.
3. Select the source language from the dropdown menu.
3. Select the target language from the dropdown menu.
4. Click the "Translate" button.
""")

# Language names
st.header('Configure Translation')
#col1, col2, col3 = st.columns(3)
codes = pd.read_csv('iso-639-3.tab', sep='\t')
LNAMES = {}
for lang in LANGS:
    lang_name = codes[codes['Id'] == lang]['Ref_Name'].values[0]
    LNAMES[lang_name] = lang

# Form inputs
if 'output_data' not in st.session_state:
    st.session_state['output_data'] = pd.DataFrame([], columns=['text', 
                                                                'translation', 'score'])
uploaded_file = st.file_uploader("Choose a *.txt file to translate:", 
                                 type="txt")
source = st.selectbox('Source Language:', list(LNAMES.keys()), index=43)
target = st.selectbox('Target Language:', list(LNAMES.keys()))
model = st.selectbox('Model:', list(MODELS.keys()))
df_view = st.radio('Sort translation output:', options=[
        'sequentially', 'by quality score (low to high quality)'], horizontal=True)

if uploaded_file and source and target and st.button('Translate!'):

    df = process_file(uploaded_file)

    if len(df) > 500:
        st.error('Please upload a file with less than 500 lines. Contact dan_whitenack@sil.org or michael_martin@sil.org if you need to enable higher volume.')
        st.stop()
    else:

        # Tyndale data
        #df = pd.read_csv('BookIntroSummaries.csv')
        #df = df[['title', 'body/p/2/__text']]
        #df.rename(columns={'body/p/2/__text': 'summary'}, inplace=True)

        # Current time in unix time as an integer
        now = int(time.time())

        # Record the translation request in the database
        cur = conn.cursor()
        print("inserting")
        cur.execute("INSERT INTO translations (source, target, num_samples, requested) VALUES (%s, %s, %s, %s)",
                    (LNAMES[source], LNAMES[target], str(len(df)), str(now)))
        conn.commit()
        cur.close()

        inputs = []
        for _, row in df.iterrows():
            source_code = LNAMES[source]
            target_code = LNAMES[target]
            text = row['text']
            inputs.append(Input(source=source_code, target=target_code, text=text).dict())

        # Translation
        with st.spinner(text="Translating your data! This may take a few minutes..."):

            # Translation
            translations = translate(inputs)

        # Quality scores
        with st.spinner(text="Calculating quality scores. Almost there!"):
            qe_inputs = []
            for i, t in enumerate(translations):
                qe_inputs.append({
                    "source": inputs[i]['text'],
                    "translation": t
                })
            f = modal.Function.lookup("qe-comet", "Model.predict")
            scores = score(qe_inputs)

    df['translation'] = translations
    df['quality score'] = scores
    if df_view == 'by quality score (low to high quality)':
        df.sort_values(by='quality score', inplace=True, ascending=True)
    st.session_state['output_data'] = df
    st.success('Done! See your translations below.')

# Output
st.header('Output (Translations and Quality Scores)')
edited_df = st.experimental_data_editor(st.session_state['output_data'])

# Download
csv = convert_df(edited_df)
if st.download_button(
    "Download",
    csv,
    "translations.csv",
    "text/csv",
    key='download-csv'
):
    # Current time in unix time as an integer
    now = int(time.time())

    # Record the translation request in the database
    cur = conn.cursor()
    cur.execute("INSERT INTO downloads (source, target, num_samples, downloaded) VALUES (%s, %s, %s, %s)",
                (LNAMES[source], LNAMES[target], str(len(edited_df)), str(now)))
    conn.commit()
    cur.close()

# More info
st.header('More Information/ Limitations')
st.markdown("""- The machine translation functionality is being further customized and developed by SIL's Language Technology team for integration into Bible translation tools like [Scripture Forge](https://tools.bible/tools/scripture-forge-2)
- The quality scores here illustrate (very minimally) the translation quality estimation functionality being developed by SIL's Innovation Development and Experimentation (IDX) for inclusion in offerings like [AQuA](https://tools.bible/tools/augmented-quality-assessment-aqua)
- **Note** - any translations you see here should be considered as "drafts" and should be post-edited by human reviewers (e.g., translation consultants) before being published or otherwise distributed.""")

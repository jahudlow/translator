#!/bin/bash

# Authenticate Modal
modal token set --token-id $MODAL_TOKEN_ID --token-secret $MODAL_TOKEN_SECRET --env=sil-ai

# Startup the streamlit app
streamlit run app.py
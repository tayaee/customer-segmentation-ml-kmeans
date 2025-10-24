@echo off
call make-venv-uv.bat
streamlit run %~n0.py

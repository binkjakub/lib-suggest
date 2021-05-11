"""
Invokes streamlit application
Credit: https://github.com/streamlit/streamlit/issues/662#issuecomment-553356419
"""
import runpy

runpy.run_module("src.app.main", run_name="__main__", alter_sys=True)

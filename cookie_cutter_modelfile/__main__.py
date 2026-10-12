"""
Entry point for running cookie_cutter_modelfile as a module.
"""

from cookie_cutter_modelfile.cli import main

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)

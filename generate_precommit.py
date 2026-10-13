import yaml

config = {
    "repos": [
        {
            "repo": "https://github.com/psf/black",
            "rev": "24.4.2",
            "hooks": [{"id": "black", "language_version": "python3.13"}],
        },
        {
            "repo": "https://github.com/pycqa/isort",
            "rev": "5.13.2",
            "hooks": [
                {
                    "id": "isort",
                    "name": "Sort imports",
                    "additional_dependencies": ["pydantic"],
                }
            ],
        },
    ],
    "post_commit": [
        {
            "name": "pre-push",
            "description": "Check git diff for any file changes",
            "entry": "git diff --cached --quiet",
            "always_run": True,
            "pass_filenames": False,
            "stages": ["push"],
        }
    ],
}

with open(".pre-commit-config.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False, width=80)

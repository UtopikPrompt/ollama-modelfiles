with open(".pre-commit-config.yaml", "rb") as f:
    content = f.read().replace(b"\r\n", b"\n")
with open(".pre-commit-config.yaml", "wb") as f:
    f.write(content)

import io

import ruamel.yaml as yaml

# Create YAML formatter with yamlfmt settings
y = yaml.YAML()
y.preserve_quotes = True
y.indent(mapping=2, sequence=4, offset=2)
y.roundtrip = True

# Read and format config.yaml
with open('config.yaml') as f:
    content = f.read()

stream = io.StringIO(content)
y.load(stream)

with open('config.yaml', 'w') as f:
    y.dump(open('config.yaml').read(), f)

print('config.yaml formatted')

# Also format config-example.yaml
with open('config-example.yaml') as f:
    content = f.read()

stream = io.StringIO(content)
y.load(stream)

with open('config-example.yaml', 'w') as f:
    y.dump(content, f)

print('config-example.yaml formatted')

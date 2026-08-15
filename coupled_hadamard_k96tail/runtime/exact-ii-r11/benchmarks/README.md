# Exact measured benchmark source

`llm_decode_bench.py.gz` is a deterministic gzip archive of the exact
`llm_decode_bench.py` bytes used for the sealed Estonia and LAVD measurements.
The decompressed SHA-256 is
`59dd767c933e06f9724a84a8883d2aac156252dbbc279ce155658005d27424d7`.
The archive SHA-256 is
`ea6b898216417bff1b35f0522536c7c46b9ca2abb5ac340704675cd82830d520`.

Extract and verify the measured source before running it:

```bash
gzip -dc llm_decode_bench.py.gz > /tmp/llm_decode_bench.measured.py
printf '%s  %s\n' \
  59dd767c933e06f9724a84a8883d2aac156252dbbc279ce155658005d27424d7 \
  /tmp/llm_decode_bench.measured.py | sha256sum -c -
```

`llm_decode_bench.ascii.py` is a readable ASCII-normalized derivative. Its
SHA-256 is
`c4ed65e97219aefd09af2baee099b47ce0b1a00dca562a4908ed8501790499ae`.
It parses as Python but is not byte-identical to the measured source and must
not be substituted when reproducing the sealed measurements.

from setuptools import setup, Extension

module = Extension(
    "symnmf_c", # must match: import symnmf_c
    sources=["symnmfmodule.c", "symnmf.c"],
    libraries=["m"], #for sqrt/exp
)

setup(
    name="symnmf_c",
    version="1.0",
    description="C extension for SymNMF",
    ext_modules=[module],
)

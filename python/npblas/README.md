# Setup

This works with Python 3.11, because that's a version Windows suggests installing by default at the time of writing, and some things (e.g. PyTorch) apparently don't work with 3.12. (It should work with Python 3.10, but won't work with anything older because we use `numpy` 2.1.)

Ensure 

```
python -m venv npblas-env
```

Windows:
```
npblas-env\Scripts\Activate.ps1
```

Linux:
```
npblas-env\Scripts\activate
```

Then:

```
pip install -r requirements.txt
```

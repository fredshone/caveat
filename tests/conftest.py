"""
Test fixtures for `caveat` package.
These fixtures will be available in any other test module.
E.g., you can define `response` as a fixture and then use it as an input argument in `test_core.py`:
```
def test_content(response):
    assert response.content
```
"""

import pytest


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/arup-group/cookiecutter-pypackage')

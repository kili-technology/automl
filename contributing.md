# Contributing

## Development environment

To install the development environment, please follow these steps:
```bash
pip install -r requirements_dev.txt
pre-commit install
pre-commit run --all-files
```

Then, the pre-commit will run automatically when you commit.



To run the unnitests:

create a .envs file with

```
KILI_API_KEY=<your_api_key>
```

Then, run the following command:

```bash
pytest
```

The KILI_API_KEY is your api key from a KILI account.
Warning: do not commit your api key.
This configuration should also allow you to run the tests with vscode Tests Explorer.
To run some tests, you should contact a team member to allow you access to the test-project.

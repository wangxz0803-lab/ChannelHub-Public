# Development and contributions

Use a virtual environment and install `.[statistical,rt,dev]`. Run `python -m pytest -q` and `python -m ruff check .` before requesting a review.

For changes to channel axes, phase, delays, antenna ordering, pilot resources or power units, include a test against an independent equation or reference vector. A change to the export contract requires a new schema version.

Keep runtime outputs outside source control. Do not submit generated datasets, environment files, credentials, deployment addresses, site data or external binary dependencies.

The repository uses a university research license with redistribution restrictions. Obtain maintainer permission before publishing modified versions or distributing code, subject to the public-hosting-platform rights described in LICENSE. Bug reports and small descriptions of proposed changes may be submitted through Issues. Coordinate code contributions and their license terms with maintainers before submission.

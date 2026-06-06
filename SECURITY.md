# Security Policy

## Supported Versions

EggHatch-AI is a proof-of-concept project. Security fixes are applied to the `main` branch.

## Reporting a Vulnerability

Please do not open public issues for sensitive vulnerabilities.

If you find a security issue, contact the maintainer through the contact information listed on the maintainer's GitHub profile. Include:

- a short description of the issue
- steps to reproduce
- affected files or endpoints
- any relevant logs or screenshots

## Scope

Relevant reports include:

- unsafe handling of local environment variables
- dependency vulnerabilities with a practical exploit path
- unintended network calls
- unsafe file handling
- documentation that could cause users to expose secrets

Out of scope:

- issues requiring malicious local code execution
- vulnerabilities in third-party services outside this repository
- model output quality issues that are not security problems

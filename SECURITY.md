# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.x     | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in Energy-Oracle, please **do not** open a public GitHub issue.

Instead, email **devneatharva@gmail.com** with:

- A clear description of the vulnerability
- Steps to reproduce (proof-of-concept where possible)
- Potential impact and affected versions
- Any suggested mitigations

We will acknowledge receipt within **48 hours** and aim to provide a fix or mitigation within **14 days** for critical issues.

## Scope

The following are in scope:

- Authentication and authorisation bypasses in the FastAPI API
- SQL injection via ORM or raw queries
- Remote code execution through the prediction pipeline
- Sensitive data exposure in logs or API responses
- Dependency vulnerabilities with a CVSS score ≥ 7.0

The following are **out of scope**:

- Denial-of-service via the rate limiter (by design, it is permissive)
- Vulnerabilities in third-party hosted infrastructure not controlled by this project
- Issues requiring physical access to the host

## Disclosure Policy

We follow responsible disclosure: once a fix is available and deployed, we will publish an advisory in [CHANGELOG.md](CHANGELOG.md) crediting the reporter (unless anonymity is requested).

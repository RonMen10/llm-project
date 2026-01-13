Simple CI/CD pipeline that runs tests on push to GitHub.

## Structure
- `scripts/` - Training and inference scripts
- `tests/` - Unit tests
- `.github/workflows/` - GitHub Actions workflow

## Workflow
On push to main/develop branches:
1. Runs unit tests
2. Tests script syntax
3. Validates project structure
4. Checks code quality

No automatic training or deployment.
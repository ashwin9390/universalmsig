# Contributing to universalmsig

We love community contributions! Whether you are interested in adding support for a new hardware backend, registering new models, or improving the translation engine, your help makes `universalmsig` more powerful for everyone.

## 🤝 How to Contribute

### 1. Reporting Issues
Found a bug or have a feature request?
* Check the [existing issues](https://github.com/ashwin9390/universalmsig/issues) to see if it’s already being tracked.
* If not, open a new issue. Please provide a clear description, the model you were using, and the target backend.

### 2. Adding New Models (Offline Registry)
We want to keep the model list growing. To add a new model:
1. Navigate to `universalmsig/core/parser.py`.
2. Add your model to the `OFFLINE_SPECS` dictionary following the existing schema.
3. Submit a PR. Your model will immediately be available for all users in offline mode.

### 3. Implementing New Backends
Want to support a new runtime (e.g., ONNX Runtime, Mojo, etc.)?
1. **Subclass**: Create a new file in `universalmsig/backends/` and inherit from `BaseBackend` (in `base.py`).
2. **Implement**: Define `name`, `supported_precisions`, `validate()`, and `compile()`.
3. **Register**: Add your new class to the `MSigTranslator` initialization in `translator.py`.

## 🛠 Development Workflow

1. **Fork** the repository and create your feature branch: `git checkout -b feat/my-new-feature`.
2. **Setup environment**:
   ```bash
   pip install -e .

```

3. **Validate**: Run the test suite before submitting:
```bash
cd universalmsig
python tests/test_universalmsig.py -v

```


*Note: We currently have 51 tests. Please ensure all tests pass.*
4. **Commit**: Use clear, conventional commit messages (e.g., `feat: add support for XYZ model`).
5. **Pull Request**: Submit your PR to the `main` branch. Provide a brief explanation of how you tested your changes.

## 📝 Code Style & Standards

* **Keep it Pythonic**: We aim for clean, readable code with zero hard dependencies where possible.
* **Documentation**: If you add a new function or class, please include a docstring.
* **Test coverage**: If adding a feature, please include a corresponding test in the `tests/` directory to maintain our 51-test verification standard.

## 🏛 Pull Request Process

* Your PR will be reviewed by maintainers.
* We may ask for changes if the logic impacts other backends.
* Once approved, your changes will be merged into the `main` branch and included in the next version.

---

*By contributing, you agree that your contributions will be licensed under the project's [Apache License 2.0](https://github.com/ashwin9390/universalmsig/blob/main/LICENSE).*


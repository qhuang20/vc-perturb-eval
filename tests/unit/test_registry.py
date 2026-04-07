import pytest

from perteval._registry import Registry


def test_register_and_get_instance():
    """Register a concrete instance and retrieve it."""
    registry: Registry[str] = Registry("test")
    registry.register("greeting", "hello")
    assert registry.get("greeting") == "hello"


def test_register_lazy_string_and_resolve():
    """Register a lazy string reference and resolve it on get()."""
    registry: Registry = Registry("test")
    registry.register("sqrt", "math:sqrt")
    func = registry.get("sqrt")
    assert func(16) == 4.0


def test_get_unknown_key_raises():
    """Requesting an unregistered key raises KeyError."""
    registry: Registry = Registry("test")
    with pytest.raises(KeyError, match="not found in test registry"):
        registry.get("nonexistent")


def test_duplicate_register_raises():
    """Registering the same name twice raises ValueError."""
    registry: Registry = Registry("test")
    registry.register("a", "value")
    with pytest.raises(ValueError, match="already registered"):
        registry.register("a", "other")


def test_list_available():
    """list_available returns sorted list of registered names."""
    registry: Registry = Registry("test")
    registry.register("beta", "b")
    registry.register("alpha", "a")
    assert registry.list_available() == ["alpha", "beta"]

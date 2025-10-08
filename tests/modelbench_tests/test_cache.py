from modelbench.cache import MBCache, NullCache, InMemoryCache, DiskCache
from pydantic import BaseModel


class TestNullCache:
    def test_basics(self):
        c: MBCache = NullCache()
        c["a"] = 1
        assert "a" not in c

    def test_context(self):
        c = NullCache()
        with c as cache:
            cache["a"] = 1
            assert "a" not in cache
        assert "a" not in c


class TestInMemoryCache:
    def test_basics(self):
        c: MBCache = InMemoryCache()
        c["a"] = 1
        assert "a" in c
        assert c["a"] == 1

    def test_context(self):
        c = InMemoryCache()
        with c as cache:
            cache["a"] = 1
            assert "a" in cache
            assert cache["a"] == 1
        assert c["a"] == 1


class Thing(BaseModel):
    x: int
    y: str


class TestDiskCache:
    def test_basics(self, tmp_path):
        c1: MBCache = DiskCache(tmp_path)
        c1["a"] = 1
        assert "a" in c1
        assert c1["a"] == 1

        c2: MBCache = DiskCache(tmp_path)
        assert "a" in c2
        assert c2["a"] == 1

        c2["a"] = 2
        assert c1["a"] == 2

        thing = Thing(x=42, y="hello")
        c2["thing"] = thing
        assert isinstance(c2["thing"], Thing)

    def test_context(self, tmp_path):
        c = DiskCache(tmp_path)
        with c as cache:
            cache["a"] = 1
            assert "a" in cache
            assert cache["a"] == 1
        assert c["a"] == 1

    def test_as_string(self, tmp_path):
        c = DiskCache(tmp_path)
        assert str(c) == f"DiskCache({tmp_path})"

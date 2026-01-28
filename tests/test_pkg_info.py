import re
import sakura


class TestPkgInfo:
    def test_version_format(self):
        assert re.match(r"^\d+\.\d+\.\d+", sakura.__version__), (
            f"Version '{sakura.__version__}' does not match semver pattern"
        )

    def test_build_is_string(self):
        assert isinstance(sakura.__build__, str)

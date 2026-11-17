import pytest

from vproc.paths import memory_directory


@pytest.mark.parametrize("identifier", ["name\x00", "name\x1f", "name\x7f", "name\x85", "name\x9f"])
@pytest.mark.parametrize("field", ["project", "memory"])
def test_storage_names_reject_the_same_controls_as_api_filters(tmp_path, identifier, field):
    with pytest.raises(ValueError, match="Invalid"):
        memory_directory(tmp_path, identifier if field == "project" else "default",
                         identifier if field == "memory" else "meeting")


def test_storage_accepts_unicode_and_internal_dots(tmp_path):
    assert memory_directory(tmp_path, "default", "réunion..draft").name == "réunion..draft"

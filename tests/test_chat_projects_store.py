"""``projects/store.py`` and ``projects/folders.py``: the chat projects store (docs/design/chat-projects.md, 1-3).

Real conversation store on a temp file throughout. Pinned: name cleaning and case-insensitive uniqueness, folder and
import-key uniqueness, the change counter, display order, filing (membership and workspace binding in one step, a
hand-set workspace never replaced unless asked, unfiling clears only what the project set), delete (unfiles, clears
the project's bindings, deletes no file), the chat-only view (only projects with the caller's own sessions), the
forget cascade, and the managed folder (visible root, slug, collisions, mode 0700).
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from prometheus.memory.lcm_conversation_store import LCMConversationStore
from prometheus.projects.folders import make_managed_folder, projects_root, slug
from prometheus.projects.store import (
    FolderInUse,
    NameTaken,
    UnknownProject,
    WorkspaceSet,
    clean_name,
)


@pytest.fixture
def conv(tmp_path):
    return LCMConversationStore(tmp_path / "lcm.db")


@pytest.fixture
def store(conv):
    return conv.chat_projects()


def _folder(tmp_path: Path, name: str) -> str:
    p = tmp_path / "folders" / name
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


class TestNames:
    @pytest.mark.parametrize(("raw", "clean"), [
        ("  Kitchen   remodel ", "Kitchen remodel"), ("a\tb\nc", "a b c"), ("x" * 80, "x" * 60), ("   ", ""),
    ])
    def test_clean(self, raw, clean):
        assert clean_name(raw) == clean

    def test_names_are_unique_in_any_case(self, store, tmp_path):
        store.insert("Beacon", _folder(tmp_path, "a"), managed=False)
        with pytest.raises(NameTaken):
            store.insert("beacon", _folder(tmp_path, "b"), managed=False)

    def test_by_name_ignores_case_and_spacing(self, store, tmp_path):
        p = store.insert("Kitchen remodel", _folder(tmp_path, "k"), managed=False)
        assert store.by_name("  kitchen   REMODEL ").id == p.id

    def test_rename_onto_another_name_is_refused_and_own_case_change_is_not(self, store, tmp_path):
        a = store.insert("Alpha", _folder(tmp_path, "a"), managed=False)
        store.insert("Beta", _folder(tmp_path, "b"), managed=False)
        with pytest.raises(NameTaken):
            store.rename(a.id, "BETA")
        assert store.rename(a.id, "ALPHA").name == "ALPHA"

    def test_rename_an_unknown_project(self, store):
        with pytest.raises(UnknownProject):
            store.rename("cp_000000000000", "X")


class TestFoldersAndKeys:
    def test_a_folder_is_used_by_one_project(self, store, tmp_path):
        f = _folder(tmp_path, "shared")
        store.insert("One", f, managed=False)
        with pytest.raises(FolderInUse):
            store.insert("Two", f, managed=False)

    def test_import_key_finds_the_project(self, store, tmp_path):
        p = store.insert("Imported", _folder(tmp_path, "i"), managed=False, import_key="local:p123")
        assert store.by_import_key("local:p123").id == p.id
        assert store.by_import_key("local:other") is None

    def test_ids_are_opaque_and_prefixed(self, store, tmp_path):
        p = store.insert("X", _folder(tmp_path, "x"), managed=True)
        assert p.id.startswith("cp_") and len(p.id) == 15
        assert p.folder_managed is True


class TestRevAndOrder:
    def test_every_change_raises_rev(self, store, tmp_path):
        r0 = store.rev()
        p = store.insert("A", _folder(tmp_path, "a"), managed=False)
        r1 = store.rev()
        store.rename(p.id, "B")
        r2 = store.rev()
        store.file("s1", p.id)
        r3 = store.rev()
        store.file("s1", None)
        r4 = store.rev()
        store.delete(p.id)
        assert r0 < r1 < r2 < r3 < r4 < store.rev()

    def test_list_is_creation_order_with_session_counts(self, store, tmp_path):
        b = store.insert("Bravo", _folder(tmp_path, "b"), managed=False)
        a = store.insert("Alpha", _folder(tmp_path, "a"), managed=False)
        store.file("s1", a.id)
        store.file("s2", a.id)
        listed = store.all()
        assert [p.id for p in listed] == [b.id, a.id]
        assert {p.id: p.sessions for p in listed} == {b.id: 0, a.id: 2}


class TestFiling:
    def test_filing_binds_the_workspace_to_the_folder(self, store, conv, tmp_path):
        p = store.insert("A", _folder(tmp_path, "a"), managed=False)
        result = store.file("s1", p.id)
        assert result.workspace == p.folder
        assert conv.get_session_workspace("s1") == p.folder
        assert store.project_of("s1") == p.id

    def test_a_hand_set_workspace_is_not_replaced_unless_asked(self, store, conv, tmp_path):
        p = store.insert("A", _folder(tmp_path, "a"), managed=False)
        conv.set_session_workspace("s1", "/somewhere/else", set_by="rest")
        with pytest.raises(WorkspaceSet) as info:
            store.file("s1", p.id)
        assert info.value.workspace == "/somewhere/else"
        assert store.project_of("s1") is None and conv.get_session_workspace("s1") == "/somewhere/else"
        store.file("s1", p.id, replace_workspace=True)
        assert conv.get_session_workspace("s1") == p.folder

    def test_moving_between_projects_rebinds(self, store, conv, tmp_path):
        a = store.insert("A", _folder(tmp_path, "a"), managed=False)
        b = store.insert("B", _folder(tmp_path, "b"), managed=False)
        store.file("s1", a.id)
        store.file("s1", b.id)
        assert store.project_of("s1") == b.id and conv.get_session_workspace("s1") == b.folder

    def test_unfiling_clears_only_what_the_project_set(self, store, conv, tmp_path):
        p = store.insert("A", _folder(tmp_path, "a"), managed=False)
        store.file("s1", p.id)
        store.file("s1", None)
        assert conv.get_session_workspace("s1") is None and store.project_of("s1") is None
        store.file("s2", p.id)
        conv.set_session_workspace("s2", "/chosen/by/hand", set_by="rest")
        store.file("s2", None)
        assert conv.get_session_workspace("s2") == "/chosen/by/hand"

    def test_filing_is_idempotent(self, store, tmp_path):
        p = store.insert("A", _folder(tmp_path, "a"), managed=False)
        store.file("s1", p.id)
        store.file("s1", p.id)
        assert store.session_ids(p.id) == ["s1"]

    def test_filing_into_an_unknown_project(self, store):
        with pytest.raises(UnknownProject):
            store.file("s1", "cp_000000000000")

    def test_projects_of_many_sessions(self, store, tmp_path):
        p = store.insert("A", _folder(tmp_path, "a"), managed=False)
        store.file("s1", p.id)
        assert store.projects_of(["s1", "s2"]) == {"s1": p.id}


class TestDelete:
    def test_delete_unfiles_clears_its_bindings_and_keeps_every_file(self, store, conv, tmp_path):
        folder = _folder(tmp_path, "keep")
        (Path(folder) / "PROMETHEUS.md").write_text("keep me")
        p = store.insert("A", folder, managed=False)
        store.file("s1", p.id)
        store.file("s2", p.id)
        conv.set_session_workspace("s2", "/by/hand", set_by="rest")
        unfiled = store.delete(p.id)
        assert unfiled == 2
        assert store.get(p.id) is None and store.project_of("s1") is None
        assert conv.get_session_workspace("s1") is None
        assert conv.get_session_workspace("s2") == "/by/hand"
        assert (Path(folder) / "PROMETHEUS.md").read_text() == "keep me"

    def test_delete_an_unknown_project(self, store):
        with pytest.raises(UnknownProject):
            store.delete("cp_000000000000")


class TestChatOnlyView:
    def test_only_projects_with_the_callers_own_sessions(self, store, tmp_path):
        mine = store.insert("Mine", _folder(tmp_path, "m"), managed=False)
        store.insert("Theirs", _folder(tmp_path, "t"), managed=False)
        store.file("own-1", mine.id)
        store.file("other-1", mine.id)
        view = store.visible_to({"own-1", "own-2"})
        assert [(v.id, v.name, v.own_session_ids) for v in view] == [(mine.id, "Mine", ["own-1"])]

    def test_nothing_owned_sees_nothing(self, store, tmp_path):
        p = store.insert("A", _folder(tmp_path, "a"), managed=False)
        store.file("s1", p.id)
        assert store.visible_to(set()) == []


def test_forgetting_a_session_removes_its_membership(conv, store, tmp_path):
    p = store.insert("A", _folder(tmp_path, "a"), managed=False)
    store.file("s1", p.id)
    conv.purge_session("s1")
    assert store.project_of("s1") is None
    assert store.get(p.id) is not None  # the project stays


class TestManagedFolder:
    @pytest.mark.parametrize(("name", "expected"), [
        ("Kitchen remodel", "kitchen-remodel"), ("  Ünïcode & stuff!! ", "unicode-stuff"),
        ("---", "project"), ("a" * 80, "a" * 48), ("Beacon / iOS", "beacon-ios"),
    ])
    def test_slug(self, name, expected):
        assert slug(name) == expected

    def test_root_defaults_to_a_visible_home_folder(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        assert projects_root({}) == tmp_path / "prometheus-projects"
        assert projects_root({"projects": {"root": "~/elsewhere"}}) == tmp_path / "elsewhere"

    def test_made_with_mode_0700_and_a_suffix_on_collision(self, tmp_path):
        root = tmp_path / "root"
        first = make_managed_folder(root, "Kitchen remodel")
        second = make_managed_folder(root, "kitchen REMODEL")
        assert first == root / "kitchen-remodel" and second == root / "kitchen-remodel-2"
        for p in (root, first, second):
            assert stat.S_IMODE(os.stat(p).st_mode) == 0o700

    def test_an_existing_file_in_the_way_is_skipped(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        (root / "notes").write_text("a file, not a folder")
        assert make_managed_folder(root, "Notes") == root / "notes-2"


class TestLabelOnly:
    """keep_workspace: a migration files as a label only, and never changes where a conversation writes."""

    def test_label_only_leaves_no_workspace_alone(self, store, conv, tmp_path):
        p = store.insert("A", _folder(tmp_path, "a"), managed=False)
        result = store.file("s1", p.id, keep_workspace=True)
        assert store.project_of("s1") == p.id
        assert result.workspace is None and conv.get_session_workspace("s1") is None

    def test_label_only_leaves_a_hand_set_workspace_alone(self, store, conv, tmp_path):
        p = store.insert("A", _folder(tmp_path, "a"), managed=False)
        conv.set_session_workspace("s1", "/by/hand", set_by="rest")
        result = store.file("s1", p.id, keep_workspace=True)
        assert result.workspace == "/by/hand" and conv.get_session_workspace("s1") == "/by/hand"
        store.file("s1", None)
        assert conv.get_session_workspace("s1") == "/by/hand"

    def test_label_only_after_a_bind_keeps_the_projects_binding(self, store, conv, tmp_path):
        p = store.insert("A", _folder(tmp_path, "a"), managed=False)
        store.file("s1", p.id)
        store.file("s1", p.id, keep_workspace=True)
        assert conv.get_session_workspace("s1") == p.folder


class TestBindingsNameTheirProject:
    """Design re-check: a binding records which project set it; only that project clears it."""

    def test_a_label_move_then_deleting_the_old_project_clears_its_binding(self, store, conv, tmp_path):
        a = store.insert("A", _folder(tmp_path, "a"), managed=False)
        b = store.insert("B", _folder(tmp_path, "b"), managed=False)
        store.file("s1", a.id)                       # bound to A's folder
        store.file("s1", b.id, keep_workspace=True)  # now a member of B, still bound to A's folder
        store.delete(a.id)
        assert conv.get_session_workspace("s1") is None  # A's binding went with A
        assert store.project_of("s1") == b.id

    def test_unfiling_from_a_label_project_leaves_another_projects_binding(self, store, conv, tmp_path):
        a = store.insert("A", _folder(tmp_path, "a"), managed=False)
        b = store.insert("B", _folder(tmp_path, "b"), managed=False)
        store.file("s1", a.id)
        store.file("s1", b.id, keep_workspace=True)
        store.file("s1", None)
        assert conv.get_session_workspace("s1") == a.folder  # B never set it, so B's unfile leaves it

    def test_the_binding_records_the_project(self, store, conv, tmp_path):
        a = store.insert("A", _folder(tmp_path, "a"), managed=False)
        store.file("s1", a.id)
        row = conv._conn.execute("SELECT set_by FROM session_workspaces WHERE session_id = 's1'").fetchone()
        assert row[0] == f"project:{a.id}"


class TestReviewFixes:
    """Review of #932."""

    def test_a_hand_set_workspace_on_the_folder_is_not_taken_over(self, store, conv, tmp_path):
        a = store.insert("A", _folder(tmp_path, "a"), managed=False)
        conv.set_session_workspace("s1", a.folder, set_by="rest")
        store.file("s1", a.id)
        store.file("s1", None)
        assert conv.get_session_workspace("s1") == a.folder  # still the owner's hand-set binding

    def test_a_failed_write_never_rolls_back_other_work_on_the_connection(self, store, conv):
        conv._conn.execute("INSERT INTO session_pins (session_id, pinned_at) VALUES ('kept', 1.0)")
        with pytest.raises(UnknownProject):
            store.rename("cp_000000000000", "X")
        conv._conn.commit()
        assert conv.is_session_pinned("kept")

    def test_building_the_store_commits_nothing_of_anyone_elses(self, tmp_path):
        conv = LCMConversationStore(tmp_path / "fresh.db")
        conv._conn.execute("INSERT INTO session_pins (session_id, pinned_at) VALUES ('pending', 1.0)")
        conv.chat_projects()
        conv._conn.rollback()
        assert not conv.is_session_pinned("pending")

    def test_forgetting_a_filed_session_raises_rev(self, conv, store, tmp_path):
        p = store.insert("A", _folder(tmp_path, "a"), managed=False)
        store.file("s1", p.id)
        before = store.rev()
        conv.purge_session("s1")
        assert store.rev() > before

    def test_an_import_key_collision_is_its_own_error(self, store, tmp_path):
        from prometheus.projects.store import ImportKeyTaken

        store.insert("A", _folder(tmp_path, "a"), managed=False, import_key="k1")
        with pytest.raises(ImportKeyTaken):
            store.insert("B", _folder(tmp_path, "b"), managed=False, import_key="k1")

    @pytest.mark.parametrize("twin", ["Café", "Ca​fé", "CAFÉ"])
    def test_names_that_look_the_same_are_the_same(self, store, tmp_path, twin):
        store.insert("Café", _folder(tmp_path, "a"), managed=False)
        with pytest.raises(NameTaken):
            store.insert(twin, _folder(tmp_path, "b"), managed=False)

    def test_an_existing_root_keeps_its_mode(self, tmp_path):
        root = tmp_path / "code"
        root.mkdir(mode=0o755)
        os.chmod(root, 0o755)
        made = make_managed_folder(root, "X")
        assert stat.S_IMODE(os.stat(root).st_mode) == 0o755
        assert stat.S_IMODE(os.stat(made).st_mode) == 0o700

    def test_a_symlinked_root_is_refused(self, tmp_path):
        target = tmp_path / "real"
        target.mkdir()
        link = tmp_path / "link"
        link.symlink_to(target)
        with pytest.raises(ValueError):
            make_managed_folder(link, "X")
        assert list(target.iterdir()) == []

    def test_the_folder_returned_is_resolved(self, tmp_path):
        made = make_managed_folder(tmp_path / "r", "X")
        assert made == made.resolve()

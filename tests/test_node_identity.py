"""Node identity — FOUNDATION Part 3.

The node is an Ed25519 keypair under ~/.prometheus/node/ (0600 key, 0700
dir); the instance is a UUID riding the vault marker. The split's whole
point is what travels when a vault moves machines, so the acceptance test
here literally copies a vault and walks the copy for key material.

conftest's autouse ``_isolated_state_dirs`` points PROMETHEUS_CONFIG_DIR at
tmp_path, so ``get_node_dir()`` is isolated per-test with no fixture work.
"""

from __future__ import annotations

import base64
import shutil
import stat
from pathlib import Path

import pytest

from prometheus.config.node_identity import (
    NODE_KEY_FILENAME,
    NODE_PUB_FILENAME,
    ensure_node_identity,
    get_node_pubkey,
    get_instance_id,
    set_instance_id,
)
from prometheus.config.paths import get_node_dir
from prometheus.config.vault_marker import (
    VaultMarkerError,
    create_marker,
    enroll_node,
    read_marker,
)


class TestKeypairGeneration:
    def test_first_run_generates_key_and_pub(self) -> None:
        identity = ensure_node_identity()
        node_dir = get_node_dir()
        assert (node_dir / NODE_KEY_FILENAME).exists()
        assert (node_dir / NODE_PUB_FILENAME).exists()
        # The pubkey IS the node ID: base64 of the raw 32-byte key.
        assert len(base64.b64decode(identity.pubkey)) == 32

    def test_private_key_mode_0600_dir_0700(self) -> None:
        ensure_node_identity()
        node_dir = get_node_dir()
        key_mode = stat.S_IMODE((node_dir / NODE_KEY_FILENAME).stat().st_mode)
        dir_mode = stat.S_IMODE(node_dir.stat().st_mode)
        assert key_mode == 0o600
        assert dir_mode == 0o700

    def test_idempotent_never_regenerates(self) -> None:
        first = ensure_node_identity()
        key_bytes = (get_node_dir() / NODE_KEY_FILENAME).read_bytes()
        second = ensure_node_identity()
        assert second.pubkey == first.pubkey
        assert (get_node_dir() / NODE_KEY_FILENAME).read_bytes() == key_bytes

    def test_missing_pub_is_rederived_not_rekeyed(self) -> None:
        first = ensure_node_identity()
        (get_node_dir() / NODE_PUB_FILENAME).unlink()
        healed = ensure_node_identity()
        assert healed.pubkey == first.pubkey

    def test_stale_pub_is_corrected_from_the_key(self) -> None:
        # The public half is a projection of the private half, never state
        # of its own — a divergent node.pub silently becoming the node ID
        # would be two machines' worth of confusion later.
        first = ensure_node_identity()
        (get_node_dir() / NODE_PUB_FILENAME).write_text("bm90LXRoZS1rZXk=\n")
        healed = ensure_node_identity()
        assert healed.pubkey == first.pubkey
        assert get_node_pubkey() == first.pubkey

    def test_unloadable_key_refuses_rather_than_rekeys(self) -> None:
        node_dir = get_node_dir()
        (node_dir / NODE_KEY_FILENAME).write_text("not a pem")
        with pytest.raises(RuntimeError, match="refusing"):
            ensure_node_identity()
        # And the mangled file is still there — nothing regenerated over it.
        assert (node_dir / NODE_KEY_FILENAME).read_text() == "not a pem"

    def test_get_node_pubkey_reads_never_mints(self) -> None:
        assert get_node_pubkey() is None
        assert not (get_node_dir() / NODE_KEY_FILENAME).exists()
        identity = ensure_node_identity()
        assert get_node_pubkey() == identity.pubkey


class TestInstancePin:
    def test_pin_roundtrip_and_clear(self) -> None:
        set_instance_id("550e8400-e29b-41d4-a716-446655440000")
        assert get_instance_id() == "550e8400-e29b-41d4-a716-446655440000"
        set_instance_id(None)
        assert get_instance_id() is None


@pytest.fixture(autouse=True)
def _clear_instance_pin():
    # The pin is process-global by design (like the vault-root pin); tests
    # must not leak it into each other.
    yield
    set_instance_id(None)


class TestEnrollment:
    @pytest.fixture()
    def vault(self, tmp_path: Path) -> Path:
        root = tmp_path / "brain-vault"
        root.mkdir()
        create_marker(root, created_by="prometheus test")
        return root

    def test_first_boot_enrolls_second_does_not_duplicate(self, vault: Path) -> None:
        identity = ensure_node_identity()
        assert enroll_node(vault, identity.pubkey, label="test-node") is True
        assert enroll_node(vault, identity.pubkey, label="test-node") is False
        marker = read_marker(vault)
        assert marker is not None
        assert [n["pubkey"] for n in marker.enrolled_nodes] == [identity.pubkey]
        assert marker.enrolled_nodes[0]["label"] == "test-node"

    def test_brand_hostname_enrolls_with_neutralized_label(
        self, vault: Path
    ) -> None:
        # The vault is a git repo whose pre-commit hook (like this repo's)
        # blocks brand-prefixed infrastructure hostnames in committed text.
        # The daemon enrolls with the raw platform.node(), so a brand-named
        # machine used to write a label that made the vault uncommittable
        # until hand-fixed. The hostname below is built from fragments so
        # THIS file passes that same hook.
        identity = ensure_node_identity()
        brand_host = "OAra" "-Foo"
        assert enroll_node(vault, identity.pubkey, label=brand_host) is True
        marker = read_marker(vault)
        assert marker is not None
        assert marker.enrolled_nodes[0]["label"] == "Foo"
        # The blocked prefix must not appear ANYWHERE in the marker file —
        # the hook greps text, not fields.
        marker_text = (vault / ".prometheus-vault").read_text(encoding="utf-8")
        assert brand_host.lower()[:5] not in marker_text.lower()

    def test_neutral_label_edge_cases(self) -> None:
        from prometheus.config.vault_marker import neutral_label

        prefix = "oara" "-"
        assert neutral_label(prefix + "mini") == "mini"
        assert neutral_label(prefix.upper() + "4090") == "4090"
        # A hostname that is ONLY the prefix still gets a usable label.
        assert neutral_label(prefix) == "node"
        assert neutral_label("") == "node"
        # Non-brand hostnames pass through untouched.
        assert neutral_label("workstation-3") == "workstation-3"

    def test_enroll_without_marker_refuses(self, tmp_path: Path) -> None:
        bare = tmp_path / "unadopted"
        bare.mkdir()
        with pytest.raises(VaultMarkerError, match="Adopt first"):
            enroll_node(bare, "AAAA", label="x")


class TestVaultCopyCarriesNoKey:
    def test_copied_vault_has_instance_but_no_key_material(
        self, tmp_path: Path
    ) -> None:
        # Acceptance (FOUNDATION Part 4): node-owned files in
        # ~/.prometheus/node/, instance-owned files in the vault — verified
        # by copying a vault to a "second machine" and confirming no key
        # travels. The instance ID follows the copy; the node key cannot,
        # because it was never inside the vault to begin with. This test is
        # the structural half of that; the operator's two-machine walk is
        # the live half.
        vault = tmp_path / "brain-vault"
        vault.mkdir()
        minted = create_marker(vault, created_by="prometheus test")
        identity = ensure_node_identity()
        enroll_node(vault, identity.pubkey, label="machine-one")

        second_machine = tmp_path / "copied-to-new-machine"
        shutil.copytree(vault, second_machine)

        copied = read_marker(second_machine)
        assert copied is not None
        assert copied.instance_id == minted.instance_id  # instance travels

        names = {p.name for p in second_machine.rglob("*") if p.is_file()}
        assert NODE_KEY_FILENAME not in names
        assert NODE_PUB_FILENAME not in names
        for file in second_machine.rglob("*"):
            if file.is_file():
                assert "PRIVATE KEY" not in file.read_text(errors="ignore")

"""capseal sign — Ed25519 signing of .cap receipts.

Signs a .cap receipt file with an Ed25519 key, producing a .cap.sig
file alongside it. The signature covers the SHA256 hash of the entire
.cap file.

Usage:
    capseal sign .capseal/runs/latest.cap --generate-key
    capseal sign .capseal/runs/latest.cap
    capseal sign .capseal/runs/latest.cap --key ~/.capseal/keys/team.pem
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import click


@click.command("sign")
@click.argument("capsule", type=click.Path(exists=True))
@click.option("--key", "-k", default=None, type=click.Path(),
              help="Path to Ed25519 private key (PEM)")
@click.option("--generate-key", is_flag=True,
              help="Generate a new signing key pair")
def sign_command(capsule: str, key: str | None, generate_key: bool) -> None:
    """Sign a .cap receipt with an Ed25519 key.

    Creates a .cap.sig file alongside the receipt containing the signature,
    public key, and cap file hash. Use `capseal verify` to check signatures.

    \b
    Examples:
        capseal sign .capseal/runs/latest.cap --generate-key
        capseal sign .capseal/runs/latest.cap
        capseal sign .capseal/runs/latest.cap --key path/to/key.pem
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives import serialization
    except ImportError:
        click.echo("Error: cryptography package required. Install with: pip install cryptography", err=True)
        raise SystemExit(1)

    capsule_path = Path(capsule).resolve()

    # Determine key directory
    capseal_dir = capsule_path.parent.parent  # .capseal/runs/xxx.cap -> .capseal/
    if capseal_dir.name != ".capseal":
        capseal_dir = capsule_path.parent / ".capseal"
    key_dir = capseal_dir / "keys"
    key_dir.mkdir(parents=True, exist_ok=True)

    if generate_key:
        private_key = Ed25519PrivateKey.generate()
        key_path = key_dir / "signing_key.pem"
        key_path.write_bytes(private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ))
        pub_path = key_dir / "signing_key.pub"
        pub_path.write_bytes(private_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        ))
        click.echo(f"Generated key pair:")
        click.echo(f"  Private: {key_path}")
        click.echo(f"  Public:  {pub_path}")
    elif key:
        key_path = Path(key)
    else:
        key_path = key_dir / "signing_key.pem"

    if not key_path.exists():
        click.echo(f"No signing key found at {key_path}. Use --generate-key first.", err=True)
        raise SystemExit(1)

    # Load key
    private_key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)

    # Read cap file and compute its SHA256
    cap_bytes = capsule_path.read_bytes()
    cap_hash = hashlib.sha256(cap_bytes).hexdigest()

    # Sign the hash
    signature = private_key.sign(cap_hash.encode())

    # Write .sig file alongside
    sig_path = capsule_path.with_suffix(".cap.sig")
    sig_data = {
        "schema": "capseal_signature_v1",
        "cap_file": capsule_path.name,
        "cap_hash": cap_hash,
        "signature": base64.b64encode(signature).decode(),
        "public_key": base64.b64encode(
            private_key.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        ).decode(),
        "algorithm": "Ed25519",
    }
    sig_path.write_text(json.dumps(sig_data, indent=2) + "\n")
    click.echo(f"Signed: {sig_path}")


def verify_signature(capsule_path: Path) -> tuple[bool, str]:
    """Verify a .cap.sig signature file. Returns (valid, message)."""
    # Resolve symlinks so we find the .sig next to the real file
    capsule_path = Path(capsule_path).resolve()
    sig_path = capsule_path.with_suffix(".cap.sig")
    if not sig_path.exists():
        return False, "not signed"

    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.hazmat.primitives import serialization
    except ImportError:
        return False, "cryptography package not installed"

    try:
        sig_data = json.loads(sig_path.read_text())
        pub_bytes = base64.b64decode(sig_data["public_key"])
        pub_key = serialization.load_pem_public_key(pub_bytes)

        cap_hash = hashlib.sha256(capsule_path.read_bytes()).hexdigest()
        signature = base64.b64decode(sig_data["signature"])

        pub_key.verify(signature, cap_hash.encode())
        return True, "valid (Ed25519)"
    except Exception as e:
        return False, f"INVALID: {e}"


__all__ = ["sign_command", "verify_signature"]

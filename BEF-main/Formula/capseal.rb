# Homebrew formula for CapSeal
# Install: brew install capseal/tap/capseal
# Or locally: brew install --build-from-source ./Formula/capseal.rb

class Capseal < Formula
  include Language::Python::Virtualenv

  desc "Cryptographic receipt generation and verification for reproducible computations"
  homepage "https://capseal.dev"
  url "https://github.com/capseal/capseal/archive/refs/tags/v0.2.0.tar.gz"
  sha256 "PLACEHOLDER_SHA256"  # Update with actual SHA256 on release
  license "MIT"
  head "https://github.com/capseal/capseal.git", branch: "main"

  depends_on "python@3.11"

  # Optional sandbox dependencies
  depends_on "bubblewrap" => :optional  # Linux-only sandbox

  resource "click" do
    url "https://files.pythonhosted.org/packages/96/d3/f04c7bfcf5c1862a2a5b845c6b2b360488cf47af55dfa79c98f6a6bf98b5/click-8.1.7.tar.gz"
    sha256 "ca9853ad459e787e2192211578cc907e7594e294c7ccc834310722b41b9ca6de"
  end

  def install
    virtualenv_install_with_resources
  end

  def caveats
    <<~EOS
      CapSeal has been installed!

      Quick start:
        capseal init          # Initialize workspace
        capseal demo          # Run self-test
        capseal doctor <file> # Verify a capsule

      For sandbox support on Linux, install bubblewrap:
        brew install bubblewrap

      Documentation: https://capseal.dev/docs
    EOS
  end

  test do
    # Run the demo command to verify installation
    system bin/"capseal", "demo", "--json"

    # Initialize a workspace
    system bin/"capseal", "init", "--json"
    assert_predicate testpath/".capseal/config.json", :exist?

    # Run happy path
    system bin/"capseal", "demo", "-o", testpath/".capseal/receipts/test.json"
    assert_predicate testpath/".capseal/receipts/test.json", :exist?
  end
end
